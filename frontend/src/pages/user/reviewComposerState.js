export function isComposerLocked({ isEditMode = false, replyToReviewId = null } = {}) {
  return Boolean(isEditMode || replyToReviewId);
}

export function getComposerMode({ isEditMode = false, replyToReviewId = null } = {}) {
  if (replyToReviewId) return "reply";
  if (isEditMode) return "edit";
  return "create";
}
