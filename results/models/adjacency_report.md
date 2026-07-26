# District adjacency

Version: `adjacency-v1`

Queen contiguity over the GADM level 1 polygons, indexed by `node_id`
0-24 from `nodes.csv` and matched on `gadm_gid`.

## Summary

| Property | Value |
| --- | --- |
| Nodes | 25 |
| Edges | 57 |
| Mean degree | 4.56 |
| Min degree | 1 |
| Max degree | 9 |
| Density | 0.190 |
| Contiguity tolerance | 100 m |
| Mean neighbour distance | 60.9 km |
| Max neighbour distance | 119.2 km |
| Gaussian sigma | 50 km |
| Gaussian edges | 53 |

## Isolated node fallback

Not used. Every district shares a boundary with at least one other.

## Neighbours

| node_id | District | Degree | Neighbours |
| --- | --- | --- | --- |
| 0 | Ampara | 6 | Badulla, Batticaloa, Hambantota, Matale, Moneragala, Polonnaruwa |
| 1 | Anuradhapura | 7 | Kurunegala, Mannar, Matale, Polonnaruwa, Puttalam, Trincomalee, Vavuniya |
| 2 | Badulla | 6 | Ampara, Kandy, Matale, Moneragala, Nuwara Eliya, Ratnapura |
| 3 | Batticaloa | 3 | Ampara, Polonnaruwa, Trincomalee |
| 4 | Colombo | 4 | Gampaha, Kalutara, Kegalle, Ratnapura |
| 5 | Galle | 3 | Kalutara, Matara, Ratnapura |
| 6 | Gampaha | 4 | Colombo, Kegalle, Kurunegala, Puttalam |
| 7 | Hambantota | 4 | Ampara, Matara, Moneragala, Ratnapura |
| 8 | Jaffna | 1 | Kilinochchi |
| 9 | Kalutara | 3 | Colombo, Galle, Ratnapura |
| 10 | Kandy | 5 | Badulla, Kegalle, Kurunegala, Matale, Nuwara Eliya |
| 11 | Kegalle | 6 | Colombo, Gampaha, Kandy, Kurunegala, Nuwara Eliya, Ratnapura |
| 12 | Kilinochchi | 3 | Jaffna, Mannar, Mullaitivu |
| 13 | Kurunegala | 6 | Anuradhapura, Gampaha, Kandy, Kegalle, Matale, Puttalam |
| 14 | Mannar | 5 | Anuradhapura, Kilinochchi, Mullaitivu, Puttalam, Vavuniya |
| 15 | Matale | 6 | Ampara, Anuradhapura, Badulla, Kandy, Kurunegala, Polonnaruwa |
| 16 | Matara | 3 | Galle, Hambantota, Ratnapura |
| 17 | Moneragala | 4 | Ampara, Badulla, Hambantota, Ratnapura |
| 18 | Mullaitivu | 4 | Kilinochchi, Mannar, Trincomalee, Vavuniya |
| 19 | Nuwara Eliya | 4 | Badulla, Kandy, Kegalle, Ratnapura |
| 20 | Polonnaruwa | 5 | Ampara, Anuradhapura, Batticaloa, Matale, Trincomalee |
| 21 | Puttalam | 4 | Anuradhapura, Gampaha, Kurunegala, Mannar |
| 22 | Ratnapura | 9 | Badulla, Colombo, Galle, Hambantota, Kalutara, Kegalle, Matara, Moneragala, Nuwara Eliya |
| 23 | Trincomalee | 5 | Anuradhapura, Batticaloa, Mullaitivu, Polonnaruwa, Vavuniya |
| 24 | Vavuniya | 4 | Anuradhapura, Mannar, Mullaitivu, Trincomalee |

## Output files

- `data/processed/adjacency.npz`

Keys: `A_binary`, `A_norm`, `A_gaussian`, `A_gaussian_norm`,
`A_distance_km`, `node_id`, `canonical_name`.
