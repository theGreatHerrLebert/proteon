# MMseqs2 On-Disk DB Format Spec

Derived from reading `src/commons/DBReader.{h,cpp}`, `DBWriter.{h,cpp}`, `FileUtil.cpp`, `Parameters.h`
on upstream rev `d45e0c4`. Target: a Rust port that round-trips byte-for-byte against upstream
`createdb` output.

## File set produced by `createdb <fasta> <outPrefix>`

For prefix `P`:

| File | Role | Format |
|---|---|---|
| `P` | data blob (sequences) | concatenated payloads, each ending `\n\0` |
| `P.index` | data index | ASCII, one line per entry: `key\toffset\tlength\n` |
| `P.dbtype` | data dbtype | 4 bytes, little-endian int32 (see bit layout below) |
| `P.lookup` | key → accession map | ASCII: `key\taccession\tfileNumber\n` |
| `P.source` | file number → source filename | ASCII: `fileNumber\tfilename\n` |
| `P_h` | header blob (FASTA descriptions) | same as `P`, parallel entries |
| `P_h.index` | header index | same as `P.index` |
| `P_h.dbtype` | header dbtype | 4 bytes; always `DBTYPE_GENERIC_DB` (12) |

Suffix convention: the "headers DB" is always prefix + `_h`. Source and lookup only exist on the data DB, not the header DB.

## `dbtype` (4 bytes LE int32)

Written little-endian regardless of host: on big-endian hosts upstream does `__builtin_bswap32` before `fwrite` (`DBWriter.cpp:193–213`), on LE hosts it's a direct write. The value is effectively always LE on disk.

Bit layout:

```
bit 31     : compression flag (1 = zstd-compressed payloads in data blob)
bits 30..17: extended flags (mask 0x7FFE when shifted; see below)
bits 16..0 : base dbtype (see enum below)
```

Read helpers: `DBReader::getExtendedDbtype(dbtype) = ((uint32_t)dbtype >> 16) & 0x7FFE`.
Write helpers: `setExtendedDbtype(dbtype, ext) = dbtype | ((ext & 0x7FFE) << 16)`.

### Base dbtype enum (`Parameters.h:69–89`)

| Value | Name | Notes |
|---|---|---|
| 0 | AMINO_ACIDS | raw protein seqs (`createdb` default on protein FASTA) |
| 1 | NUCLEOTIDES | raw nt seqs |
| 2 | HMM_PROFILE | profile DB |
| 3 | *(reserved)* | formerly PROFILE_STATE_SEQ |
| 4 | *(reserved)* | formerly PROFILE_STATE_PROFILE |
| 5 | ALIGNMENT_RES | alignment results |
| 6 | CLUSTER_RES | clustering results |
| 7 | PREFILTER_RES | prefilter hits |
| 8 | TAXONOMICAL_RESULT | |
| 9 | INDEX_DB | precomputed k-mer index |
| 10 | CA3M_DB | compressed A3M |
| 11 | MSA_DB | MSAs |
| 12 | GENERIC_DB | arbitrary bytes (headers use this) |
| 13 | OMIT_FILE | sentinel; writer skips .dbtype |
| 14 | PREFILTER_REV_RES | |
| 15 | OFFSETDB | |
| 16–20 | DIRECTORY / FLATFILE / SEQTAXDB / STDIN / URI | CLI verification only, not real dbtypes on disk |

### Extended flags (`Parameters.h:91–95`)

Stored shifted into bits 17..30:

| Bit (pre-shift) | Name | Meaning |
|---|---|---|
| 1 | EXTENDED_COMPRESSED | redundant with top bit, legacy compat |
| 2 | EXTENDED_INDEX_NEED_SRC | index references source file |
| 4 | EXTENDED_CONTEXT_PSEUDO_COUNTS | |
| 8 | EXTENDED_GPU | padded/GPU-ready layout (written by `makepaddedseqdb`) |
| 16 | EXTENDED_SET | proteome-set DB (multihit) |

For the phase-1 port, treat dbtype as opaque: read 4 bytes LE int32, write 4 bytes LE int32. No interpretation needed for round-trip.

## `.index` (ASCII)

One record per line:

```
<key>\t<offset>\t<length>\n
```

- `key`: unsigned 32-bit integer, decimal (`u32toa_sse2` in `DBWriter.cpp:420`)
- `offset`: unsigned 64-bit integer, decimal, byte offset into data blob
- `length`: unsigned 64-bit integer, decimal, byte length of payload **including** trailing `\n\0`

Index is typically (but not required to be) sorted by key ascending. `readIndex` tracks `isSortedById` and falls back to linear scan if not sorted.

For the `examples/DB.fasta` reference DB: first entry `0\t0\t1882`, last `19999\t9095261\t308`, sum `9095261+308 = 9095569 = file size` — verified.

## Data blob

Concatenation of per-entry payloads. Each payload stored at `index[i].offset` of length `index[i].length`. Each payload ends with `\n\0`:

- the `\n` is an upstream convention (created by `createdb` appending a newline to each record)
- the `\0` is the null terminator `DBWriter` writes at `writeEnd` (`DBWriter.cpp:436`)
- both are included in the `length` field

For compressed DBs (top bit of dbtype set): each payload is prefixed by a 4-byte `unsigned int compressedLengthInt` holding the *uncompressed* size, followed by zstd-compressed bytes. `DBWriter.cpp:361`. For phase 1 (uncompressed round-trip), ignore this.

## `.lookup` (ASCII)

One record per line:

```
<key>\t<accession>\t<fileNumber>\n
```

- `key`: u32 decimal, matches data `.index` keys
- `accession`: FASTA accession parsed from description line (e.g. `W0FSK4`, `Q2NHD6`)
- `fileNumber`: u32 decimal, index into `.source` (0 when only one FASTA input)

Sort order: by id (key) by default on read (`DBReader.cpp:152`), also supports sort-by-accession mode.

## `.source` (ASCII)

One record per source FASTA:

```
<fileNumber>\t<filename>\n
```

For `createdb examples/DB.fasta targetDB`: single line `0\tDB.fasta\n` (11 bytes).

## Endianness and portability

- `.dbtype` is LE int32 on disk (normalized via bswap on BE hosts).
- All other files (`.index`, `.lookup`, `.source`, data blob) are ASCII or raw bytes — endian-independent.
- The data blob itself is raw payload bytes, so endianness of downstream consumers doesn't matter for round-trip.

## What to port vs what to skip for phase 1

**Port (byte-exact round-trip target):**
- Reading `.dbtype` as LE int32 + parsing base/extended bits
- Reading `.index` (ASCII parse: key, offset, length)
- Reading `.lookup` + `.source`
- Reading data blob via mmap + slicing by `(offset, length)`
- Writing symmetric `DBWriter` for all of the above

**Skip for phase 1:**
- Compression path (zstd prefix-length encoding) — add in phase 1.5 once uncompressed works
- GPU/padded layout (`EXTENDED_GPU`, `makepaddedseqdb`) — phase 4 concern
- Per-thread index files + merge (upstream parallelizes writes by thread; the merged `.index` is a simple concat — our single-threaded port produces byte-identical output for free)
- Multi-file (split-file) DBs — `createdb` on a single input FASTA doesn't produce these

## Acceptance gate (phase 1)

```
upstream: mmseqs createdb examples/DB.fasta /tmp/ref/targetDB
ours:     cargo run -- createdb-rs examples/DB.fasta /tmp/ours/targetDB

for f in targetDB targetDB.index targetDB.dbtype targetDB.lookup \
         targetDB.source targetDB_h targetDB_h.index targetDB_h.dbtype; do
  cmp /tmp/ref/$f /tmp/ours/$f
done
```

All 8 files byte-identical → phase 1 done.

**Secondary gate:** read the upstream DB with our `DBReader`, write it back with our `DBWriter`,
cmp against upstream — this validates reader and writer independently catch the same format.
