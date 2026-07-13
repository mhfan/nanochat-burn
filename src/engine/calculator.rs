
/// Safe handcrafted mathematical and string expression evaluator in pure Rust.
/// Supports basic arithmetic (+, -, *, /, parentheses) and ".count()" string queries.
pub fn use_calculator(expr: &str) -> Option<String> {
    let expr_trimmed = expr.trim();

    // 1. String count matching: e.g. "strawberry".count("r")
    if expr_trimmed.contains(".count(") &&
        let Some(res) = parse_and_eval_count(expr_trimmed) {
        return Some(res.to_string());
    }

    // 2. Arithmetic expression evaluation
    parse_and_eval_arithmetic(&expr_trimmed.replace(',', "")).filter(|value| value.is_finite())
        .map(|value| if value.fract() == 0.0 {
            (value as i64).to_string()
        } else { value.to_string() })
}

fn parse_and_eval_count(s: &str) -> Option<usize> {
    let count_marker = ".count(";
    let idx = s.find(count_marker)?;
    let prefix = s[..idx].trim();
    let suffix = s[idx + count_marker.len()..].trim();

    if !suffix.ends_with(')') { return None; }
    let arg = suffix[..suffix.len() - 1].trim();

    let main_str = extract_quoted_string(prefix)?;
    let sub_str = extract_quoted_string(arg)?;

    if sub_str.is_empty() { return Some(main_str.chars().count() + 1); }
    Some(main_str.matches(&sub_str).count())
}

fn extract_quoted_string(s: &str) -> Option<String> {
    s.strip_prefix('"').and_then(|s| s.strip_suffix('"'))
        .or_else(|| s.strip_prefix('\'').and_then(|s| s.strip_suffix('\'')))
        .map(str::to_string)
}

struct Parser<'a> {
    input: &'a str,
    pos: usize,
}

impl<'a> Parser<'a> {
    fn new(input: &'a str) -> Self { Parser { input, pos: 0 } }
    fn peek(&self) -> Option<char> { self.input[self.pos..].chars().next() }

    fn consume(&mut self) -> Option<char> {
        let c = self.peek()?;
        self.pos += c.len_utf8();
        Some(c)
    }

    fn skip_whitespace(&mut self) {
        while let Some(c) = self.peek() {
            if c.is_whitespace() { self.consume(); } else { break; }
        }
    }

    fn parse_expression(&mut self) -> Option<f64> {
        self.skip_whitespace();
        let mut val = self.parse_term()?;

        loop {
            self.skip_whitespace();
            match self.peek() {
                Some('+') => {
                    self.consume();
                    let right = self.parse_term()?;
                    val += right;
                }
                Some('-') => {
                    self.consume();
                    let right = self.parse_term()?;
                    val -= right;
                }
                _ => break,
            }
        }
        Some(val)
    }

    fn parse_term(&mut self) -> Option<f64> {
        self.skip_whitespace();
        let mut val = self.parse_factor()?;

        loop {
            self.skip_whitespace();
            match self.peek() {
                Some('*') => {
                    self.consume();
                    let right = self.parse_factor()?;
                    val *= right;
                }
                Some('/') => {
                    self.consume();
                    let right = self.parse_factor()?;
                    if right == 0.0 { return None; }
                    val /= right;
                }
                _ => break,
            }
        }
        Some(val)
    }

    fn parse_factor(&mut self) -> Option<f64> {
        self.skip_whitespace();
        match self.peek()? {
            '(' => {
                self.consume();
                let val = self.parse_expression()?;
                self.skip_whitespace();
                if self.peek() == Some(')') {
                    self.consume();
                    Some(val)
                } else { None }
            }
            c if c.is_ascii_digit() || c == '.' || c == '-' || c == '+' => {
                let start = self.pos;
                if c == '-' || c == '+' { self.consume(); }
                let mut has_digits = false;
                while let Some(next_c) = self.peek() {
                    if next_c.is_ascii_digit() {
                        self.consume();
                        has_digits = true;
                    } else if next_c == '.' {
                        self.consume();
                    } else { break; }
                }
                if !has_digits { return None; }
                let num_str = &self.input[start..self.pos];
                num_str.trim().parse::<f64>().ok()
            }
            _ => None,
        }
    }
}

fn parse_and_eval_arithmetic(s: &str) -> Option<f64> {
    if !s.chars().all(|c| c.is_ascii_digit() || "+-*/.() ".contains(c)) { return None; }
    let mut parser = Parser::new(s);
    let val = parser.parse_expression()?;
    parser.skip_whitespace();
    if parser.pos == s.len() { Some(val) } else { None }
}

#[cfg(test)] mod tests { use super::*;
    #[test] fn test_calculator_arithmetic() {
        assert_eq!(use_calculator("5 + 3 * 2"), Some("11".to_string()));
        assert_eq!(use_calculator("(5 + 3) * 2"), Some("16".to_string()));
        assert_eq!(use_calculator("10 / 2"), Some("5".to_string()));
        assert_eq!(use_calculator("10 / 0"), None);
        assert_eq!(use_calculator("invalid_chars + 5"), None);
    }

    #[test] fn test_calculator_string_count() {
        assert_eq!(use_calculator("\"strawberry\".count(\"r\")"), Some("3".to_string()));
        assert_eq!(use_calculator("'banana'.count('an')"), Some("2".to_string()));
        assert_eq!(use_calculator("\"hello\".count(\"l\")"), Some("2".to_string()));
        assert_eq!(use_calculator("\"a,b\".count(\",\")"), Some("1".to_string()));
        assert_eq!(use_calculator("\"你好\".count(\"\")"), Some("3".to_string()));
    }
}
