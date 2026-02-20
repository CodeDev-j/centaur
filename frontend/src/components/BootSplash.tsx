/**
 * Boot splash overlay. The `run-splash` class is set on <html> in layout.tsx,
 * so CSS animations trigger on every page load. The overlay auto-hides via
 * animation-fill-mode: forwards.
 */
export default function BootSplash() {
  return (
    <div
      id="boot-splash"
      aria-hidden="true"
      className="boot-splash-overlay"
    >
      <span className="boot-splash-wordmark text-h1">CENTAUR</span>
    </div>
  );
}
