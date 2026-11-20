# parlaspeech-forced-alignment

## v0.1
- focus on HR RS SI
- take only instances without numerals and without brackets `[({})]`
- select N hours of data (N to be discussed with "a pal")
- do two rounds
1. build a model and apply it on your data
2. select only the "well-performing" data (discussion with "my pal" or discuss anew) for the second training round
- align everything, keep only stuff above a threshold (TBD)   
