# IPTA Data Combination v1.07

| Status:   | Complete |
| --------- | -------- |
| Date complete: | 27.11.2015 |
| Author(s):| Joris Verbiest |
| Contact:  | verbiest@physik.uni-bielefeld.de |
| Title:    | Notes on the IPTA data combination v1.07 |
| Keywords: | IPTA Data combination 1, data, processing |

## Changes from the previous data combination:

  * Combinations A, B: The Nancay ToAs for PSR J1713+0747 were not updated in Combinations A and B. This has now been fixed. 
  * Combination B: DM Corrections in Combination B for PSR J1713+0747 Nancay data were missing for 5 ToAs. This has now been fixed.
  * Combinations A, B, C: For pulsars with no DM model (beyond DM2) and with insignificant DM1 and insignificant DM2, these parameters (DM1,DM2) have been removed (again) from the par file.

## Following discussions at the IPTA meeting in Banff, we present three releases: 

  * Version A ("raw"): The raw par and tim files, without EFACs/EQUADs, without DM corrections, without red noise models.
  * Version B ("classic"): The traditional combination, with T2EFACs/T2EQUADs per ToA group; a Cholesky prewhitening file per pulsar; and DM corrections implemented through -dmo flags in the tim files (but without uncertainties in the DM model).
  * Version C ("New"): A new combination, with TNEFACs/TNEQUADs and TempoNest-based red noise and DM models. 


