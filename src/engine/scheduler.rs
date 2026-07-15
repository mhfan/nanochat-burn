use std::collections::{BTreeMap, VecDeque};

pub type RequestId = u64;

#[derive(Debug)]
struct Request<T> { id: RequestId, payload: T }

#[derive(Debug)]
pub struct ContinuousBatchScheduler<T> {
    capacity: usize,
    next_id: RequestId,
    waiting: VecDeque<Request<T>>,
    active: BTreeMap<RequestId, T>,
}

impl<T> ContinuousBatchScheduler<T> {
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "scheduler capacity must be positive");
        Self { capacity, next_id: 0, waiting: VecDeque::new(), active: BTreeMap::new() }
    }

    pub fn submit(&mut self, payload: T) -> RequestId {
        let id = self.next_id;
        self.next_id = self.next_id.checked_add(1).expect("request ID overflow");
        self.waiting.push_back(Request { id, payload });
        id
    }

    pub fn schedule_iteration(&mut self) -> Vec<RequestId> {
        while self.active.len() < self.capacity {
            let Some(request) = self.waiting.pop_front() else { break; };
            self.active.insert(request.id, request.payload);
        }
        self.active.keys().copied().collect()
    }

    pub fn get(&self, id: RequestId) -> Option<&T> { self.active.get(&id) }
    pub fn get_mut(&mut self, id: RequestId) -> Option<&mut T> { self.active.get_mut(&id) }

    pub fn complete(&mut self, id: RequestId) -> Option<T> { self.active.remove(&id) }

    pub fn cancel(&mut self, id: RequestId) -> Option<T> {
        if let Some(payload) = self.active.remove(&id) { return Some(payload); }
        let index = self.waiting.iter().position(|request| request.id == id)?;
        self.waiting.remove(index).map(|request| request.payload)
    }

    pub fn active_len(&self) -> usize { self.active.len() }
    pub fn waiting_len(&self) -> usize { self.waiting.len() }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttentionSinkPolicy { pub sink_tokens: usize, pub recent_tokens: usize }

impl AttentionSinkPolicy {
    pub fn retained_ranges(self, sequence_len: usize) -> Vec<std::ops::Range<usize>> {
        let sink_end = self.sink_tokens.min(sequence_len);
        let recent_start = sequence_len.saturating_sub(self.recent_tokens).max(sink_end);
        let mut ranges = Vec::with_capacity(2);
        if sink_end > 0 { ranges.push(0..sink_end); }
        if recent_start < sequence_len { ranges.push(recent_start..sequence_len); }
        ranges
    }
}

#[cfg(test)] mod tests { use super::*;
    #[test] fn test_continuous_batch_admission_completion_and_cancel() {
        let mut scheduler = ContinuousBatchScheduler::new(2);
        let (first, second, third) =
            (scheduler.submit("a"), scheduler.submit("b"), scheduler.submit("c"));
        assert_eq!(scheduler.schedule_iteration(), vec![first, second]);
        assert_eq!(scheduler.waiting_len(), 1);
        assert_eq!(scheduler.complete(first), Some("a"));
        assert_eq!(scheduler.schedule_iteration(), vec![second, third]);
        assert_eq!(scheduler.cancel(second), Some("b"));
        assert_eq!(scheduler.cancel(third), Some("c"));
        assert_eq!(scheduler.active_len(), 0);
    }

    #[test] fn test_attention_sink_ranges() {
        let policy = AttentionSinkPolicy { sink_tokens: 4, recent_tokens: 8 };
        assert_eq!(policy.retained_ranges(20), vec![0..4, 12..20]);
        assert_eq!(policy.retained_ranges(6), vec![0..4, 4..6]);
    }
}
