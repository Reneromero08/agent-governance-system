// Small debounce utility. Returns a function that delays calling `fn`
// until `delay` ms after the last call.

export function debounce(fn, delay) {
  let t = null;
  const wrapped = (...args) => {
    if (t) clearTimeout(t);
    t = setTimeout(() => { t = null; fn(...args); }, delay);
  };
  wrapped.flush = () => {
    if (t) { clearTimeout(t); t = null; fn(); }
  };
  wrapped.cancel = () => {
    if (t) { clearTimeout(t); t = null; }
  };
  return wrapped;
}
