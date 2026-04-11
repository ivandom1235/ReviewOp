export const DEFAULT_SEARCH_SORT = "most_recent";
export const DEFAULT_SEARCH_MIN_RATING = 0;

export function getSearchState(params) {
  return {
    q: params.get("q") || "",
    minRating: Number(params.get("min_rating") ?? String(DEFAULT_SEARCH_MIN_RATING)),
    sort: params.get("sort") || DEFAULT_SEARCH_SORT,
  };
}

export function hasSearchFilters({ q = "", minRating = DEFAULT_SEARCH_MIN_RATING, sort = DEFAULT_SEARCH_SORT } = {}) {
  return Boolean(q || Number(minRating) > DEFAULT_SEARCH_MIN_RATING || sort !== DEFAULT_SEARCH_SORT);
}

export function updateSearchParams(currentParams, next) {
  const current = new URLSearchParams(currentParams);
  Object.entries(next).forEach(([key, value]) => {
    if (value === "" || value == null) {
      current.delete(key);
    } else {
      current.set(key, String(value));
    }
  });
  return current;
}

export function resetSearchResultsState() {
  return {
    rows: [],
    hasMore: true,
    error: "",
    searchInput: "",
  };
}
