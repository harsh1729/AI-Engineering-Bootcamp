/** Must match backend `GUEST_USAGE_LIMIT_MESSAGE` in app/config.py */
export const GUEST_USAGE_LIMIT_MESSAGE =
  "Guest user limit reached. Register for more messages.";

/** Must match backend `LOGGED_IN_USER_USAGE_LIMIT_MESSAGE` in app/config.py */
export const LOGGED_IN_USER_USAGE_LIMIT_MESSAGE =
  "Account message limit reached. Please try again later.";

/** Set when the register page exists (e.g. "/register"). */
export const REGISTER_PATH = "/register";

export function isGuestUsageLimitMessage(message) {
  return message === GUEST_USAGE_LIMIT_MESSAGE;
}

export function isLoggedInUserUsageLimitMessage(message) {
  return message === LOGGED_IN_USER_USAGE_LIMIT_MESSAGE;
}
