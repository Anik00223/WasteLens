# Iteration 10: Fresh-Set Error Taxonomy & Label Audit

## 1. Summary of Supported Fresh Set (63 images)
- Total supported items: 63
- Correct: 25 (39.68%)
- Errors: 38 (60.32%)

## 2. Error Breakdown by Taxonomy Category
| Error Category | Count | Primary Mechanism |
|---|---:|---|
| **background/context** | 16 | Primary failure mode for category |
| **packaging variation** | 11 | Primary failure mode for category |
| **genuine classifier confusion** | 6 | Primary failure mode for category |
| **class ambiguity** | 5 | Primary failure mode for category |

## 3. Confusion Pairs (True Class -> Predicted Class)
| True Class -> Predicted Class | Count | Key Driver |
|---|---:|---|
| `organic -> recyclable` | 15 | Domain shift toward dominant class |
| `hazardous -> recyclable` | 10 | Domain shift toward dominant class |
| `general trash -> recyclable` | 10 | Domain shift toward dominant class |
| `recyclable -> hazardous` | 1 | Domain shift toward dominant class |
| `hazardous -> organic` | 1 | Domain shift toward dominant class |
| `general trash -> hazardous` | 1 | Domain shift toward dominant class |

## 4. Full Error Table (38 misclassifications)
| Image | Category | True Label | Predicted | Top P | Runner-Up (P) | Reject P | Verdict | Error Category |
|---|---|---|---|---:|---|---:|---|---|
| `supported-organic__Compost__00.jpg` | Compost | **organic** | `recyclable` | 1.000 | organic (0.000) | 0.2918 | unsupported | background/context |
| `supported-organic__Compost__01.jpg` | Compost | **organic** | `recyclable` | 1.000 | hazardous (0.000) | 0.5338 | unsupported | background/context |
| `supported-organic__Compost__02.jpg` | Compost | **organic** | `recyclable` | 1.000 | organic (0.000) | 0.0812 | unsupported | background/context |
| `supported-organic__Compost__03.jpg` | Compost | **organic** | `recyclable` | 1.000 | organic (0.000) | 0.9976 | unsupported | background/context |
| `supported-organic__Compost__04.jpg` | Compost | **organic** | `recyclable` | 1.000 | organic (0.000) | 0.0001 | supported | background/context |
| `supported-organic__Compost__05.jpg` | Compost | **organic** | `recyclable` | 1.000 | hazardous (0.000) | 0.9997 | unsupported | background/context |
| `supported-organic__Food_waste__00.jpg` | Food waste | **organic** | `recyclable` | 0.782 | organic (0.218) | 0.9803 | unsupported | genuine classifier confusion |
| `supported-organic__Food_waste__01.jpg` | Food waste | **organic** | `recyclable` | 1.000 | organic (0.000) | 0.4240 | unsupported | genuine classifier confusion |
| `supported-organic__Food_waste__03.jpg` | Food waste | **organic** | `recyclable` | 1.000 | organic (0.000) | 0.0000 | supported | genuine classifier confusion |
| `supported-organic__Food_waste__04.jpg` | Food waste | **organic** | `recyclable` | 1.000 | organic (0.000) | 0.0008 | supported | genuine classifier confusion |
| `supported-organic__Food_waste__05.jpg` | Food waste | **organic** | `recyclable` | 1.000 | organic (0.000) | 0.0000 | supported | genuine classifier confusion |
| `supported-organic__Compost_bins__00.jpg` | Compost bins | **organic** | `recyclable` | 1.000 | organic (0.000) | 0.7968 | unsupported | background/context |
| `supported-organic__Compost_bins__01.jpg` | Compost bins | **organic** | `recyclable` | 0.998 | organic (0.002) | 0.1276 | unsupported | background/context |
| `supported-organic__Compost_bins__02.jpg` | Compost bins | **organic** | `recyclable` | 1.000 | organic (0.000) | 0.0051 | supported | background/context |
| `supported-organic__Compost_bins__03.jpg` | Compost bins | **organic** | `recyclable` | 1.000 | organic (0.000) | 0.9962 | unsupported | background/context |
| `supported-recyclable__Aluminium_cans__03.jpg` | Aluminium cans | **recyclable** | `hazardous` | 0.998 | recyclable (0.002) | 0.0179 | supported | genuine classifier confusion |
| `supported-hazardous__Lead-acid_batteries__02.jpg` | Lead-acid batteries | **hazardous** | `recyclable` | 0.987 | hazardous (0.013) | 0.0039 | supported | packaging variation |
| `supported-hazardous__Button_cells__00.jpg` | Button cells | **hazardous** | `recyclable` | 0.924 | organic (0.066) | 0.0001 | supported | packaging variation |
| `supported-hazardous__Button_cells__01.jpg` | Button cells | **hazardous** | `organic` | 0.670 | general trash (0.226) | 0.9901 | unsupported | packaging variation |
| `supported-hazardous__Button_cells__02.jpg` | Button cells | **hazardous** | `recyclable` | 1.000 | organic (0.000) | 0.0000 | supported | packaging variation |
| `supported-hazardous__Button_cells__03.jpg` | Button cells | **hazardous** | `recyclable` | 0.999 | hazardous (0.001) | 0.0000 | supported | packaging variation |
| `supported-hazardous__Button_cells__04.jpg` | Button cells | **hazardous** | `recyclable` | 1.000 | hazardous (0.000) | 0.0000 | supported | packaging variation |
| `supported-hazardous__Fluorescent_lamps__00.jpg` | Fluorescent lamps | **hazardous** | `recyclable` | 1.000 | hazardous (0.000) | 0.0016 | supported | packaging variation |
| `supported-hazardous__Fluorescent_lamps__01.jpg` | Fluorescent lamps | **hazardous** | `recyclable` | 1.000 | hazardous (0.000) | 0.0000 | supported | packaging variation |
| `supported-hazardous__Fluorescent_lamps__02.jpg` | Fluorescent lamps | **hazardous** | `recyclable` | 1.000 | hazardous (0.000) | 0.9364 | unsupported | packaging variation |
| `supported-hazardous__Fluorescent_lamps__03.jpg` | Fluorescent lamps | **hazardous** | `recyclable` | 1.000 | hazardous (0.000) | 0.0000 | supported | packaging variation |
| `supported-hazardous__Fluorescent_lamps__04.jpg` | Fluorescent lamps | **hazardous** | `recyclable` | 1.000 | hazardous (0.000) | 0.8582 | unsupported | packaging variation |
| `supported-general__Litter__00.jpg` | Litter | **general trash** | `recyclable` | 1.000 | general trash (0.000) | 0.3785 | unsupported | background/context |
| `supported-general__Litter__01.jpg` | Litter | **general trash** | `recyclable` | 1.000 | hazardous (0.000) | 0.0351 | supported | background/context |
| `supported-general__Litter__02.jpg` | Litter | **general trash** | `recyclable` | 1.000 | organic (0.000) | 0.0298 | supported | background/context |
| `supported-general__Litter__03.jpg` | Litter | **general trash** | `recyclable` | 1.000 | hazardous (0.000) | 0.0931 | unsupported | background/context |
| `supported-general__Litter__04.jpg` | Litter | **general trash** | `recyclable` | 1.000 | organic (0.000) | 0.8999 | unsupported | background/context |
| `supported-general__Litter__05.jpg` | Litter | **general trash** | `recyclable` | 1.000 | organic (0.000) | 0.4085 | unsupported | background/context |
| `supported-general__Paper_towels__00.jpg` | Paper towels | **general trash** | `recyclable` | 1.000 | organic (0.000) | 0.9636 | unsupported | class ambiguity |
| `supported-general__Paper_towels__01.jpg` | Paper towels | **general trash** | `recyclable` | 1.000 | organic (0.000) | 0.0075 | supported | class ambiguity |
| `supported-general__Paper_towels__02.jpg` | Paper towels | **general trash** | `recyclable` | 1.000 | hazardous (0.000) | 0.0000 | supported | class ambiguity |
| `supported-general__Paper_towels__03.jpg` | Paper towels | **general trash** | `recyclable` | 1.000 | organic (0.000) | 0.9983 | unsupported | class ambiguity |
| `supported-general__Paper_towels__04.jpg` | Paper towels | **general trash** | `hazardous` | 1.000 | recyclable (0.000) | 0.0000 | supported | class ambiguity |
