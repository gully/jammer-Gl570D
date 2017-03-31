data
---

This directory contains "**raw**" and **reduced** subdirectories.

raw is a little misleading in this context.  The data in the raw directory is fully reduced and flux calibrated, it's just the poor naming convention I chose to discriminate between data that is-or-is-not prepared for ingestion into Starfish.  

The reduced directory contains spectra that are prepared for ingestion into Starfish.  These files have the requisite units and file structures.

The `.pic` files are Python binary pickle files, the `.hdf5` files are binary HDF5 files.

This data is *not* currently committed to the repository.  To acquire this data, you have to email a request to Mike Line, then process the files with the example Jupyter notebooks.

```bash
├── raw
│   ├── 2M_J0050.pic
│   ├── 2M_J0415.pic
│   ├── 2M_J0727.pic
│   ├── 2M_J0729.pic
│   ├── 2M_J0939.pic
│   ├── 2M_J1114.pic
│   ├── 2M_J1217.pic
│   ├── 2M_J1553.pic
│   ├── Gl570D.pic
│   ├── HD3651B.pic
│   └── SDSS_1416b.pic
└── reduced
    ├── 2M_J0050.hdf5
    ├── 2M_J0415.hdf5
    ├── 2M_J0727.hdf5
    ├── 2M_J0729.hdf5
    ├── 2M_J0939.hdf5
    ├── 2M_J1114.hdf5
    ├── 2M_J1217.hdf5
    ├── 2M_J1553.hdf5
    ├── Gl570D.hdf5
    ├── HD3651B.hdf5
    └── SDSS_1416b.hdf5
```
