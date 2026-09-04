# Images

`banner.svg` is the source of truth. Everything else is generated from it — edit
the SVG and regenerate, never touch the raster files by hand.

| file | size | use |
|---|---|---|
| `banner.svg` | vector | **source.** Edit this one. |
| `banner.webp` | 2752×1536 | README and web — smallest for the same pixels |
| `banner.png` | 2752×1536 | blog posts, docs, anything that will not take WebP |
| `social-preview.png` | 1280×640 | GitHub repo social preview (GitHub's own spec) |
| `og-image.png` | 1200×630 | Open Graph / Twitter cards (the OG spec size) |

**PNG exists because WebP does not work for social cards.** Open Graph scrapers
and several platforms silently fail on WebP, and a card that fails renders as a
blank rectangle rather than an error — so it is easy to ship broken and never
know. The two card sizes are genuinely different specs, not a rounding: GitHub
wants 1280×640, Open Graph wants 1200×630.

## Regenerating

```bash
cd images
rsvg-convert -w 2752 -h 1536 banner.svg -o banner.png
magick banner.png -quality 92 banner.webp

rsvg-convert -w 1280 banner.svg -o /tmp/sp.png
magick /tmp/sp.png -gravity center -crop 1280x640+0+0 +repage -quality 92 social-preview.png

rsvg-convert -w 1200 banner.svg -o /tmp/og.png
magick /tmp/og.png -gravity center -crop 1200x630+0+0 +repage -quality 95 og-image.png
```

The crops trim the white margin rather than the artwork — the SVG is 1.79:1 and
both card specs are wider, so rendering to width and cropping height centred is
what keeps the card from letterboxing.

Every file must stay under **500 KB**: the repo's pre-commit hook
(`check-added-large-files --maxkb=500`) rejects anything larger.
