# PyPI Publishing Setup

## 1. Register packages on PyPI

- Go to https://pypi.org/manage/account/publishing/
- Add pending publisher for `ferritin-connector`:
  - Owner: `theGreatHerrLebert`
  - Repository: `ferritin`
  - Workflow: `release.yml`
  - Environment: `pypi`
- Add pending publisher for `ferritin`:
  - Same settings as above

## 2. Create GitHub environment

- Go to repo Settings → Environments → New environment → name it `pypi`
- No special rules needed (trusted publishing handles auth)

## 3. Protect main branch (optional but recommended)

- Settings → Branches → Add branch protection rule
- Branch name pattern: `main`
- Check: "Require a pull request before merging"
- Check: "Require status checks to pass" → select `rust-tests` and `python-tests`

## 4. First release

```bash
# Tag a release
git tag v0.1.0
git push origin v0.1.0
```

- Then go to GitHub → Releases → Create release from tag `v0.1.0`
- The CI will automatically:
  1. Build wheels for Linux (x86_64 + aarch64), macOS (Intel + Apple Silicon), Windows
  2. Publish `ferritin-connector` to PyPI
  3. Publish `ferritin` to PyPI (after connector is live)

## 5. Verify

```bash
pip install ferritin
python -c "import ferritin; print(ferritin.__version__)"
```
