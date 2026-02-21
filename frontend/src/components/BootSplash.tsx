/**
 * Boot splash overlay — plays on every page load (1000ms).
 *
 * The inline <script> runs synchronously before first paint,
 * adding `run-splash` to <html> so the CSS animation fires immediately.
 * CSS handles the animation (blur-in → hold → dissolve) via the `run-splash` parent selector.
 */
export default function BootSplash() {
  return (
    <>
      <script
        dangerouslySetInnerHTML={{
          __html: `(function(){try{document.documentElement.classList.add("run-splash")}catch(e){}})()`,
        }}
      />
      <div id="boot-splash" aria-hidden="true" className="boot-splash-overlay">
        <span className="boot-splash-wordmark text-display text-[var(--text-primary)]">
          CENTAUR
        </span>
      </div>
    </>
  );
}
