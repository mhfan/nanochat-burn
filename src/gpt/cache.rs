use burn::tensor::{Bool, Int, Shape, Tensor, backend::Backend};

use super::repeat_kv;

const DEFAULT_PAGE_SIZE: usize = 8;

#[derive(Clone, Debug)]
pub struct PageAllocator {
    free: Vec<usize>,
    allocated: Vec<bool>,
}

impl PageAllocator {
    pub fn new(num_pages: usize) -> Self {
        Self { free: (0..num_pages).rev().collect(), allocated: vec![false; num_pages] }
    }

    pub fn allocate(&mut self) -> Option<usize> {
        let page = self.free.pop()?;
        assert!(!self.allocated[page], "free list contains an allocated page");
        self.allocated[page] = true;
        Some(page)
    }

    pub fn release(&mut self, page: usize) {
        assert!(self.allocated.get(page).copied().unwrap_or(false),
            "cannot release an unallocated page");
        self.allocated[page] = false;
        self.free.push(page);
    }

    pub fn available(&self) -> usize { self.free.len() }
    pub fn capacity(&self) -> usize { self.allocated.len() }
}

pub trait KvCacheControl {
    fn len(&self) -> usize;
    fn capacity(&self) -> usize;
    fn truncate(&mut self, len: usize);
    fn release_request(&mut self, request: usize);
    fn block_table(&self, request: usize) -> &[Option<usize>];
}

#[derive(Clone, Debug)]
pub struct KVCache<B: Backend> {
    // Layer -> [max_num_pages, page_size, n_kv_head, head_dim]
    pub k_page_pool: Vec<Tensor<B, 4>>,
    // Layer -> [max_num_pages, page_size, n_kv_head, head_dim]
    pub v_page_pool: Vec<Tensor<B, 4>>,
    pub page_size: usize,
    pub max_pages_per_seq: usize,
    pub batch_size: usize,
    pub max_seq_len: usize,
    pub token_history: Option<Tensor<B, 2, Int>>,
    pub len: usize,
    pub request_lens: Vec<usize>,
    pub allocator: PageAllocator,
    pub block_tables: Vec<Vec<Option<usize>>>,
}

impl<B: Backend> KVCache<B> {
    pub fn new_allocated(n_layer: usize, batch_size: usize, max_seq_len: usize,
        n_kv_head: usize, head_dim: usize, device: &B::Device) -> Self {
        Self::new_paged(n_layer, batch_size, max_seq_len, n_kv_head,
            head_dim, DEFAULT_PAGE_SIZE, device)
    }

    pub fn new_contiguous(n_layer: usize, batch_size: usize, max_seq_len: usize,
        n_kv_head: usize, head_dim: usize, device: &B::Device) -> Self {
        Self::new_paged(n_layer, batch_size, max_seq_len, n_kv_head,
            head_dim, max_seq_len, device)
    }

    pub fn new_paged(n_layer: usize, batch_size: usize, max_seq_len: usize,
        n_kv_head: usize, head_dim: usize, page_size: usize, device: &B::Device) -> Self {
        assert!(n_layer > 0, "cache must contain at least one layer");
        assert!(batch_size > 0, "cache batch size must be greater than zero");
        assert!(max_seq_len > 0, "cache sequence length must be greater than zero");
        assert!(n_kv_head > 0 && head_dim > 0, "cache head dimensions must be non-zero");
        assert!(page_size > 0, "page size must be greater than zero");
        let max_pages_per_seq = max_seq_len.div_ceil(page_size);
        let max_num_pages = batch_size * max_pages_per_seq;

        let pool_shape = Shape::new([max_num_pages, page_size, n_kv_head, head_dim]);
        let (k_page_pool, v_page_pool) = (0..n_layer).map(|_| {(
                    Tensor::zeros(pool_shape.clone(), device),
                    Tensor::zeros(pool_shape.clone(), device),
            )}).unzip();

        Self { k_page_pool, v_page_pool, page_size, max_pages_per_seq,
            batch_size, max_seq_len, token_history: None, len: 0,
            request_lens: vec![0; batch_size],
            allocator: PageAllocator::new(max_num_pages),
            block_tables: vec![vec![None; max_pages_per_seq]; batch_size],
        }
    }

    #[cfg(test)]
    pub(super) fn ensure_pages(&mut self, start_page: usize, end_page: usize) {
        for request in 0..self.batch_size {
            self.ensure_request_pages(request, start_page, end_page);
        }
    }

    fn ensure_request_pages(&mut self, request: usize, start_page: usize, end_page: usize) {
        for logical_page in start_page..=end_page {
            if self.block_tables[request][logical_page].is_none() {
                let page = self.allocator.allocate().unwrap_or_else(||
                    panic!("KV page pool exhausted while extending request {request}"));
                self.block_tables[request][logical_page] = Some(page);
            }
        }
    }

    pub fn truncate_request(&mut self, request: usize, len: usize) {
        assert!(request < self.batch_size, "request index exceeds cache batch size");
        let retained_pages = len.div_ceil(self.page_size);
        for logical_page in retained_pages..self.max_pages_per_seq {
            if let Some(page) = self.block_tables[request][logical_page].take() {
                self.allocator.release(page);
            }
        }
        if let Some(history) = self.token_history.take() {
            let history = if len < self.max_seq_len {
                let device = history.device();
                history.slice_assign([request..request + 1, len..self.max_seq_len],
                    Tensor::zeros([1, self.max_seq_len - len], &device))
            } else { history };
            self.token_history = Some(history);
        }
        self.request_lens[request] = len;
        self.len = self.request_lens.iter().copied().max().unwrap_or(0);
    }

    pub fn truncate(&mut self, len: usize) {
        assert!(len <= self.len, "cannot extend a cache by truncating it");
        for request in 0..self.batch_size {
            self.truncate_request(request, len.min(self.request_lens[request]));
        }
    }

    pub fn release_request(&mut self, request: usize) {
        self.truncate_request(request, 0);
    }

    pub fn evict_for_attention_sinks(&mut self, request: usize, sink_tokens: usize,
        recent_tokens: usize) {
        assert!(request < self.batch_size, "request index exceeds cache batch size");
        let len = self.request_lens[request];
        let sink_pages = sink_tokens.min(len).div_ceil(self.page_size);
        let recent_start = len.saturating_sub(recent_tokens) / self.page_size;
        for logical_page in sink_pages..recent_start {
            if let Some(page) = self.block_tables[request][logical_page].take() {
                self.allocator.release(page);
            }
        }
    }

    pub fn block_table(&self, request: usize) -> &[Option<usize>] {
        self.block_tables.get(request).map(Vec::as_slice)
            .unwrap_or_else(|| panic!("request index exceeds cache batch size"))
    }

    pub(super) fn update(&mut self, layer_idx: usize, k: Tensor<B, 4>, v: Tensor<B, 4>,
        step: usize) {
        let [batch_size, seq_len, n_kv_head, head_dim] = k.shape().dims();
        self.validate_update(layer_idx, &v, [batch_size, seq_len, n_kv_head, head_dim]);
        assert_eq!(batch_size, self.batch_size, "cache batch size mismatch");
        for request in 0..batch_size {
            self.update_request(layer_idx, &k, &v, request, request, step,
                seq_len, n_kv_head, head_dim);
        }
    }

    pub(super) fn update_rows(&mut self, layer_idx: usize, k: Tensor<B, 4>, v: Tensor<B, 4>,
        requests: &[usize], steps: &[usize]) {
        let [source_batch_size, seq_len, n_kv_head, head_dim] = k.shape().dims();
        self.validate_update(layer_idx, &v,
            [source_batch_size, seq_len, n_kv_head, head_dim]);
        assert_eq!(requests.len(), source_batch_size, "cache request mapping size mismatch");
        assert_eq!(steps.len(), source_batch_size, "cache position mapping size mismatch");
        assert!(requests.iter().all(|&request| request < self.batch_size),
            "cache request index exceeds batch capacity");

        for (source_batch_idx, (&request, &step)) in
            requests.iter().zip(steps).enumerate() {
            self.update_request(layer_idx, &k, &v, source_batch_idx, request, step,
                seq_len, n_kv_head, head_dim);
        }
    }

    fn validate_update(&self, layer_idx: usize, v: &Tensor<B, 4>, shape: [usize; 4]) {
        assert_eq!(v.shape().dims(), shape, "key/value cache shape mismatch");
        assert!(layer_idx < self.k_page_pool.len() && layer_idx < self.v_page_pool.len(),
            "cache does not contain model layer {layer_idx}");
    }

    #[allow(clippy::too_many_arguments)]
    fn update_request(&mut self, layer_idx: usize, k: &Tensor<B, 4>, v: &Tensor<B, 4>,
        source_batch_idx: usize, request: usize, step: usize, seq_len: usize,
        n_kv_head: usize, head_dim: usize) {
        let end = step.checked_add(seq_len).expect("cache position overflow");
        assert!(end <= self.max_seq_len, "cached sequence exceeds cache capacity");
        self.request_lens[request] = self.request_lens[request].max(end);
        self.len = self.len.max(end);
        let (start_page, end_page) = (step / self.page_size, (end - 1) / self.page_size);
        self.ensure_request_pages(request, start_page, end_page);
        for page in start_page..=end_page {
            let (pos_start, pos_end) =
                (step.max(page * self.page_size), end.min((page + 1) * self.page_size));
            let (token_start, token_end) = (pos_start - step, pos_end - step);
            let offset_start = pos_start % self.page_size;
            let offset_end = offset_start + token_end - token_start;
            let physical_page = self.block_tables[request][page]
                .expect("logical KV page was not allocated");
            let source = [source_batch_idx..source_batch_idx + 1, token_start..token_end,
                0..n_kv_head, 0..head_dim];
            let target = [physical_page..physical_page + 1, offset_start..offset_end,
                0..n_kv_head, 0..head_dim];
            let shape = [1, token_end - token_start, n_kv_head, head_dim];
            self.k_page_pool[layer_idx] = self.k_page_pool[layer_idx].clone()
                .slice_assign(target.clone(), k.clone().slice(source.clone()).reshape(shape));
            self.v_page_pool[layer_idx] = self.v_page_pool[layer_idx].clone()
                .slice_assign(target, v.clone().slice(source).reshape(shape));
        }
    }

    pub(super) fn attend(&self, layer_idx: usize, q: Tensor<B, 4>, mask: Tensor<B, 4, Bool>,
        end: usize, n_head: usize, n_kv_head: usize, head_dim: usize) -> Tensor<B, 4> {
        let [batch_size, query_len, _, _] = q.shape().dims();
        let group_size = n_head / n_kv_head;
        let num_pages = end.div_ceil(self.page_size);
        let mut outputs = Vec::with_capacity(batch_size);

        for request in 0..batch_size {
            let query = q.clone().slice([request..request + 1, 0..query_len,
                0..n_head, 0..head_dim]).swap_dims(1, 2);
            let mut page_scores = Vec::with_capacity(num_pages);
            let mut page_values = Vec::with_capacity(num_pages);
            let mut page_maxima = Vec::with_capacity(num_pages);
            for logical_page in 0..num_pages {
                let Some(physical) = self.block_tables[request][logical_page] else { continue; };
                let key_start = logical_page * self.page_size;
                let valid = (end - key_start).min(self.page_size);
                let key = self.k_page_pool[layer_idx].clone().slice([
                    physical..physical + 1, 0..valid, 0..n_kv_head, 0..head_dim]);
                let value = self.v_page_pool[layer_idx].clone().slice([
                    physical..physical + 1, 0..valid, 0..n_kv_head, 0..head_dim]);
                let key = repeat_kv(key, group_size).swap_dims(1, 2);
                let value = repeat_kv(value, group_size).swap_dims(1, 2);
                let scores = (query.clone().matmul(key.swap_dims(2, 3)) *
                    (1.0 / (head_dim as f32).sqrt())).mask_fill(
                    mask.clone().slice([0..1, 0..1, 0..query_len,
                        key_start..key_start + valid]), f32::NEG_INFINITY);
                page_maxima.push(scores.clone().max_dim(3));
                page_scores.push(scores);
                page_values.push(value);
            }
            let global_max = Tensor::cat(page_maxima, 3).max_dim(3);
            let mut numerator: Option<Tensor<B, 4>> = None;
            let mut denominator: Option<Tensor<B, 4>> = None;
            for (scores, value) in page_scores.into_iter().zip(page_values) {
                let weights = (scores - global_max.clone()).exp();
                let page_numerator = weights.clone().matmul(value);
                let page_denominator = weights.sum_dim(3);
                numerator = Some(numerator.map_or(page_numerator.clone(),
                    |sum| sum + page_numerator));
                denominator = Some(denominator.map_or(page_denominator.clone(),
                    |sum| sum + page_denominator));
            }
            outputs.push((numerator.unwrap() / denominator.unwrap().clamp(1e-12, 1e12))
                .swap_dims(1, 2));
        }
        Tensor::cat(outputs, 0)
    }

    pub(super) fn attend_rows(&self, layer_idx: usize, q: Tensor<B, 4>,
        mask: Tensor<B, 4, Bool>,
        requests: &[usize], steps: &[usize], n_head: usize, n_kv_head: usize,
        head_dim: usize) -> Tensor<B, 4> {
        let [source_batch_size, query_len, _, _] = q.shape().dims();
        assert_eq!(requests.len(), source_batch_size, "cache request mapping size mismatch");
        assert_eq!(steps.len(), source_batch_size, "cache position mapping size mismatch");
        let group_size = n_head / n_kv_head;
        let mut outputs = Vec::with_capacity(source_batch_size);

        for (source_batch_idx, (&request, &step)) in
            requests.iter().zip(steps).enumerate() {
            let end = step + query_len;
            let num_pages = end.div_ceil(self.page_size);
            let query = q.clone().slice([source_batch_idx..source_batch_idx + 1, 0..query_len,
                0..n_head, 0..head_dim]).swap_dims(1, 2);
            let mut page_scores = Vec::with_capacity(num_pages);
            let mut page_values = Vec::with_capacity(num_pages);
            let mut page_maxima = Vec::with_capacity(num_pages);
            for logical_page in 0..num_pages {
                let Some(physical) = self.block_tables[request][logical_page] else { continue; };
                let key_start = logical_page * self.page_size;
                let valid = (end - key_start).min(self.page_size);
                let key = self.k_page_pool[layer_idx].clone().slice([
                    physical..physical + 1, 0..valid, 0..n_kv_head, 0..head_dim]);
                let value = self.v_page_pool[layer_idx].clone().slice([
                    physical..physical + 1, 0..valid, 0..n_kv_head, 0..head_dim]);
                let key = repeat_kv(key, group_size).swap_dims(1, 2);
                let value = repeat_kv(value, group_size).swap_dims(1, 2);
                let scores = (query.clone().matmul(key.swap_dims(2, 3)) *
                    (1.0 / (head_dim as f32).sqrt())).mask_fill(
                    mask.clone().slice([0..1, 0..1, step..end,
                        key_start..key_start + valid]), f32::NEG_INFINITY);
                page_maxima.push(scores.clone().max_dim(3));
                page_scores.push(scores);
                page_values.push(value);
            }
            let global_max = Tensor::cat(page_maxima, 3).max_dim(3);
            let mut numerator: Option<Tensor<B, 4>> = None;
            let mut denominator: Option<Tensor<B, 4>> = None;
            for (scores, value) in page_scores.into_iter().zip(page_values) {
                let weights = (scores - global_max.clone()).exp();
                let page_numerator = weights.clone().matmul(value);
                let page_denominator = weights.sum_dim(3);
                numerator = Some(numerator.map_or(page_numerator.clone(),
                    |sum| sum + page_numerator));
                denominator = Some(denominator.map_or(page_denominator.clone(),
                    |sum| sum + page_denominator));
            }
            outputs.push((numerator.unwrap() / denominator.unwrap().clamp(1e-12, 1e12))
                .swap_dims(1, 2));
        }
        Tensor::cat(outputs, 0)
    }
}

impl<B: Backend> KvCacheControl for KVCache<B> {
    fn len(&self) -> usize { self.len }
    fn capacity(&self) -> usize { self.max_seq_len }
    fn truncate(&mut self, len: usize) { KVCache::truncate(self, len); }
    fn release_request(&mut self, request: usize) { KVCache::release_request(self, request); }
    fn block_table(&self, request: usize) -> &[Option<usize>] {
        KVCache::block_table(self, request)
    }
}
