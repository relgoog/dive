use crate::dive;

#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Code {
   Ok                 = 0,
   Cancelled          = 1,
   Unknown            = 2,
   InvalidArgument    = 3,
   DeadlineExceeded   = 4,
   NotFound           = 5,
   AlreadyExists      = 6,
   PermissionDenied   = 7,
   ResourceExhausted  = 8,
   FailedPrecondition = 9,
   Aborted            = 10,
   OutOfRange         = 11,
   Unimplemented      = 12,
   Internal           = 13,
   Unavailable        = 14,
   DataLoss           = 15,
   Unauthenticated    = 16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Status {
   pub code: Code,
}

impl Status {
   #[inline]
   #[must_use]
   pub const fn ok() -> Self {
      Self { code: Code::Ok }
   }

   #[inline]
   #[must_use]
   pub const fn new(code: Code) -> Self {
      Self { code }
   }

   #[inline]
   #[must_use]
   pub const fn code(&self) -> Code {
      self.code
   }

   #[inline]
   #[must_use]
   pub const fn is_ok(&self) -> bool {
      matches!(self.code, Code::Ok)
   }
}

#[inline]
pub fn check_op_helper_out_of_line(status: &Status, expr: &str) {
   if !status.is_ok() {
      dive::log(expr, "status.cc");
      dive::cease(status.code() as i32, "status.cc");
   }
}
