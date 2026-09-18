# Terminology

- Tracked types: The kinds of memory objects that we wish to track in and report on from in the simulator.
- Provenance: Data tracking host writes, provided from the caller, can be modifed during the simulator run.
- Dense range: Equal-sized elements whose values were produced by the same update and that we can treat the same way.
- Marking: Metadata attached to a memory range identifying a tracked type.
- Pbuffer pointer: Project shorthand for a physical-storage-buffer / device-address pointer.
- Candidate / verified candidate: A suspected tracked value; verified means simulation confirmed its use as that type.
- Data source / bit component: The input bit range contributing to part of a discovered tracked value.
- Arbitrary value: A value conservatively treated as potentially having any valid value of its type.
- Thread-specific value: A value influenced by invocation/workgroup built-ins.
- Full dispatch: Simulation of every invocation is needed to guarantee complete write-side tracking.
- Lineage: Tracking identity through memory copies to allow later markings to propagate to the originating write.
- Relevant data: Any memory value that may affect tracked types.
- Side-effect: Write-out from simulator to (relevant) external data or when we find consumption of tracked types.
