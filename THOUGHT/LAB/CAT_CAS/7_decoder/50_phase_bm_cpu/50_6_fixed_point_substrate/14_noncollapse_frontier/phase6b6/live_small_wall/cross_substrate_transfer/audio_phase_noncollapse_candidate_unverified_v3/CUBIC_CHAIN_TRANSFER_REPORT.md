# Cubic Chain Transfer Report

Candidate O: F17 cubic-chain reversible transfer mechanism.

Exact family closure survived: `True`
Classification: `SOURCE_REPRODUCED_FAMILY_SCOPED_TRANSFER_CLOSURE`

V3 reconstruction actually performed:

- nodes=2: pebble_slots=2, pebble_cells=544, pebble_apps=1, two_message_cells=544
- nodes=3: pebble_slots=3, pebble_cells=816, pebble_apps=3, two_message_cells=544
- nodes=5: pebble_slots=4, pebble_cells=1088, pebble_apps=9, two_message_cells=544
- nodes=9: pebble_slots=5, pebble_cells=1360, pebble_apps=27, two_message_cells=544
- nodes=17: pebble_slots=6, pebble_cells=1632, pebble_apps=81, two_message_cells=544
- nodes=33: pebble_slots=7, pebble_cells=1904, pebble_apps=243, two_message_cells=544
- nodes=65: pebble_slots=8, pebble_cells=2176, pebble_apps=729, two_message_cells=544

Finding:

The exact topology-factorized path transfer appears to close for the declared F17 cubic-chain family and removes the explicit 17^k assignment trace for tested nodes. In V3, exact boundary and restoration parity remain source/oracle-supported rather than independently reimplemented.

Baseline discipline:

The reversible pebble schedule is not the strongest compact method. The identical exact two-message path dynamic program uses 544 integer cells and less transfer work for nodes at least 3. Integer payload width also grows with depth. Therefore this is source-reproduced family-scoped transfer closure, not an independently verified transferable reversible-pebble law or Small Wall evidence.
