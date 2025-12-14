# Bioinformatics Tools Website

A Quarto-based website for hosting bioinformatics tools, tutorials, and workshop materials. The site is published via GitHub Pages at https://alex-sandercock.github.io/biotools.

## Structure

```
biotools/
├── _quarto.yml          # Main configuration
├── index.qmd            # Home page (about the work)
├── about.qmd            # About me page
├── resources.qmd        # Curated links
├── styles.css           # Custom styling
├── tools/               # Tool documentation
│   ├── index.qmd
│   ├── tool-one.qmd
│   └── tool-two.qmd
├── tutorials/           # Self-paced tutorials
│   ├── index.qmd
│   └── tutorial-*.qmd
├── workshops/           # Workshop materials
│   ├── index.qmd
│   └── shiny-2025/      # Individual workshop
├── images/              # Images, GIFs
└── files/               # Downloadable files (data, PDFs)
```

## Getting Started

### Prerequisites

1. Install [Quarto](https://quarto.org/docs/get-started/)
2. Install R (if using R code execution)

### Local Development

```bash
# Clone the repository
git clone https://github.com/alex-sandercock/biotools.git
cd biotools

# Preview the site
quarto preview
```

The site will open in your browser at `http://localhost:4321` and auto-reload when you make changes.

### Build the Site

```bash
quarto render
```

Output goes to the `docs/` folder.

## Deploying to GitHub Pages

### Option 1: Deploy from `docs/` folder

1. In `_quarto.yml`, keep `output-dir: docs`
2. Run `quarto render`
3. Commit and push, including the `docs/` folder
4. In GitHub repo settings → Pages → Source: Deploy from branch → `main` → `/docs`

### Option 2: GitHub Actions (automated)

Create `.github/workflows/publish.yml`:

```yaml
name: Publish site

on:
  push:
    branches: [main]

jobs:
  build-deploy:
    runs-on: ubuntu-latest
    permissions:
      contents: write
    steps:
      - uses: actions/checkout@v4
      
      - uses: quarto-dev/quarto-actions/setup@v2
      
      - name: Render
        run: quarto render
        
      - name: Deploy
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./docs
```

Then set Pages source to `gh-pages` branch.

## Customization

### Adding a new tool

1. Create `tools/new-tool.qmd`
2. Add to the navbar menu in `_quarto.yml`
3. Use the existing tool pages as templates

### Adding a new tutorial

1. Create `tutorials/tutorial-XX-topic.qmd`
2. Add to the sidebar in `_quarto.yml`
3. Update `tutorials/index.qmd`

### Adding a new workshop

1. Create `workshops/workshop-name/` folder
2. Add `index.qmd`, `setup.qmd`, and module files
3. Add to sidebar in `_quarto.yml`
4. Update `workshops/index.qmd`

### Adding images and GIFs

1. Place files in `images/`
2. Reference with `![Alt text](images/filename.png)` or `![](images/demo.gif)`

### Adding downloadable files

1. Place files in `files/`
2. Link with `[Download](files/data.zip)`

## Tips

### Recording GIFs for Shiny apps

- [ScreenToGif](https://www.screentogif.com/) (Windows)
- [Gifski](https://gif.ski/) (Mac/Linux)
- Keep under 10 seconds, 10-15 fps for reasonable file sizes

### Code execution

If you want R code to execute during render:

```yaml
# In the YAML header of a .qmd file
execute:
  eval: true
```

Set `eval: false` (default in this template) to just display code without running it.

## License

Content: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
Code: [MIT](LICENSE)
