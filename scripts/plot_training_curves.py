import argparse
import html
import json
import os
import re

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


DEFAULT_TAGS = [
    "train_R_eps_avg_0",
    "train_R_eps_avg_1",
    "eval_R_eps_avg_0",
    "eval_R_eps_avg_1",
    "eval_win_rate_0",
    "eval_win_rate_1",
    "episode_length",
]


def safe_name(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def make_svg(title, points):
    width, height = 760, 300
    left, right, top, bottom = 64, 24, 28, 42
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if xmin == xmax:
        xmax = xmin + 1
    if ymin == ymax:
        ymin -= 1
        ymax += 1

    def sx(x):
        return left + (x - xmin) / (xmax - xmin) * (width - left - right)

    def sy(y):
        return top + (ymax - y) / (ymax - ymin) * (height - top - bottom)

    poly = " ".join("%.2f,%.2f" % (sx(x), sy(y)) for x, y in points)
    circles = "\n".join(
        '<circle cx="%.2f" cy="%.2f" r="3.5" fill="#2563eb"/>' % (sx(x), sy(y))
        for x, y in points
    )
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="{left}" y="20" font-family="Arial" font-size="14" fill="#111827">{html.escape(title)}</text>
  <line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#9ca3af"/>
  <line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#9ca3af"/>
  <text x="{left}" y="{height-12}" font-family="Arial" font-size="11" fill="#4b5563">epoch {xmin:g} to {xmax:g}</text>
  <text x="8" y="{top+10}" font-family="Arial" font-size="11" fill="#4b5563">{ymax:.3g}</text>
  <text x="8" y="{height-bottom}" font-family="Arial" font-size="11" fill="#4b5563">{ymin:.3g}</text>
  <polyline fill="none" stroke="#2563eb" stroke-width="2.5" points="{poly}"/>
  {circles}
</svg>
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", action="append", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--tag", action="append", default=None)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tags = args.tag or DEFAULT_TAGS
    summary = {}
    for run_dir in args.run_dir:
        tb_dir = os.path.join(run_dir, "tb")
        acc = EventAccumulator(tb_dir)
        acc.Reload()
        available = set(acc.Tags().get("scalars", []))
        run_name = os.path.basename(run_dir.rstrip(os.sep))
        summary[run_dir] = {}
        for tag in tags:
            if tag not in available:
                continue
            events = acc.Scalars(tag)
            points = [(event.step, float(event.value)) for event in events]
            if not points:
                continue
            filename = "%s__%s.svg" % (safe_name(run_name), safe_name(tag))
            path = os.path.join(args.out_dir, filename)
            with open(path, "w") as f:
                f.write(make_svg("%s / %s" % (run_name, tag), points))
            summary[run_dir][tag] = {
                "path": path,
                "points": points,
                "last": points[-1][1],
            }

    summary_path = os.path.join(args.out_dir, "curve_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
