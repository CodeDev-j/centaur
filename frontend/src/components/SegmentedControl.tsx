"use client";

interface SegmentOption {
  value: string;
  label: string;
  disabled?: boolean;
  tooltip?: string;
}

interface SegmentedControlProps {
  options: SegmentOption[];
  value: string | string[];
  onChange: (value: string) => void;
  multi?: boolean;
}

export default function SegmentedControl({
  options,
  value,
  onChange,
  multi = false,
}: SegmentedControlProps) {
  const selected = Array.isArray(value) ? value : [value];

  return (
    <div className="flex items-center gap-1">
      {options.map((opt) => {
        const isActive = selected.includes(opt.value);
        return (
          <button
            key={opt.value}
            onClick={() => onChange(opt.value)}
            disabled={opt.disabled}
            title={opt.tooltip}
            className={`px-2.5 py-1 rounded text-[10px] uppercase tracking-wider
              border transition-colors duration-75
              ${isActive
                ? "bg-[var(--bg-surface)] border-[var(--border-sharp)] text-[var(--text-primary)]"
                : "border-transparent text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-surface)]"
              }
              ${opt.disabled ? "opacity-30 cursor-not-allowed" : "cursor-pointer"}
            `}
          >
            {multi && (
              <span className={`inline-block w-1.5 h-1.5 rounded-full mr-1.5 align-middle
                ${isActive ? "bg-[var(--accent)]" : "border border-[var(--text-secondary)]"}
              `} />
            )}
            {opt.label}
          </button>
        );
      })}
    </div>
  );
}
