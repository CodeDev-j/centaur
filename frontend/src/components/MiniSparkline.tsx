/**
 * Tiny SVG sparkline for inline data trends.
 * Zero dependencies — pure math + SVG.
 */

interface MiniSparklineProps {
  values: number[];
  width?: number;
  height?: number;
}

export default function MiniSparkline({
  values,
  width = 52,
  height = 20,
}: MiniSparklineProps) {
  if (values.length < 2) return null;

  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const padY = height * 0.15;

  const points = values
    .map((v, i) => {
      const x = (i / (values.length - 1)) * width;
      const y = padY + ((max - v) / range) * (height - 2 * padY);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");

  const trend = values[values.length - 1] - values[0];
  const color = trend > 0 ? "#10b981" : trend < 0 ? "#f43f5e" : "#6b7280";

  return (
    <svg width={width} height={height} className="shrink-0">
      <polyline
        points={points}
        fill="none"
        stroke={color}
        strokeWidth="1.5"
        strokeLinejoin="round"
        strokeLinecap="round"
      />
    </svg>
  );
}
