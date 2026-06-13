# Electrostatics validation corpus

A local fixture of 317 RCSB PDB structures used to exercise the
`proteon-electrostatics` BEM port on realistic SES-fed meshes (near-singular
collocation, mesh-acceptance, scaling). See `TO_ELECTROSTATICS.md` (P6.5) for how
it is used.

**Not committed.** The `*.pdb` / `*.pqr` files (~254 MB) are git-ignored
(`/.gitignore`) — this is a local fixture, not repository content. Only this
README is tracked, so the directory and its provenance survive in git.

## Re-fetch

The structures are plain RCSB downloads. To repopulate:

```sh
# from a list of 4-letter PDB IDs (one per line) in ids.txt
while read id; do
  curl -fsSL "https://files.rcsb.org/download/${id}.pdb" -o "${id}.pdb"
done < ids.txt
```

The exact ID list is whatever currently sits in this directory; regenerate it
with `ls *.pdb | sed 's/\.pdb$//' > ids.txt` before clearing, if you need to
reproduce the same set.
