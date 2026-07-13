# Learning-Enhanced Composite DNA Data Storage Under Sampling Randomness and IDS Errors

This repository contains the official PyTorch implementation for the paper:  
**"Learning-Enhanced Composite DNA Data Storage Under Sampling Randomness and IDS Errors"**  
by *Busra Tegin* and *Tolga M. Duman*

---

## Contributing Institutions
* **CentraleSupelec**, Rennes Campus, France and IETR-UMR CNRS 6164
* **Bilkent University**, Ankara, Turkey



## Funding & Acknowledgment
This work was funded by the **European Union** through the **European Research Council (ERC)** Executive Agency under the **ERC Advanced Grant 101054904: TRANCIDS**. Views and opinions expressed are, however, those of the authors only and do not necessarily reflect those of the European Union or the European Research Council Executive Agency.



###  Note on Parametrization & Simulation Abstractions
* **`K` vs `n`:** In the provided codebase, the variable `K` represents the **number of read copies/samples per strand**, which is formally denoted as **$n$** in the text of the paper. The code's internal sequence windows (`160` states) map directly to the strand block lengths ($E$)
* **Substitution Channel Equivalence:** To facilitate streamlined tensor processing, the data generation blocks utilize a symmetric uniform 4-ary substitution model (transitions to all four bases with probability $p_s/4$) - please be careful (Figs. 5-9 states that $p_d = p_i = \frac{4}{3}p_s$ for compensation) -> This is only for the learning-based approaches - the analiytical approach uses exactly the same channel model as described in the code. 



##  Required Packages & Libraries

To run the training pipelines and test evaluators, ensure you have Python 3.8+ installed along with the following libraries:

```bash
pip install torch numpy scipy tqdm pandas openpyxl
