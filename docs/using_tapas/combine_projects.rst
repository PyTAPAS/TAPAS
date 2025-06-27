Combine Projects Tab
--------------------

The **Combine Projects** tab merges multiple datasets—each covering different spectral or temporal regions under similar experimental conditions—onto a single, unified grid.  After selecting the source datasets, the user chooses overlap-handling and gap-extrapolation rules.  TAPAS then builds the merged data matrix, saves it as a new root dataset in a separate HDF5 project, and records links to all parent projects.  The resulting project offers a coherent data surface for downstream analysis while preserving full provenance of every original measurement.  
