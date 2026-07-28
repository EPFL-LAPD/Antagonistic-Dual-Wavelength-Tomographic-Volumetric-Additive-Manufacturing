
<!-- PROJECT LOGO -->
<br />
<p align="center">

  <h1 align="center"><a href="https://chemrxiv.org/doi/abs/10.26434/chemrxiv.15006647/v1">Antagonistic Dual-Wavelength Tomographic Volumetric Additive Manufacturing</a></h1>

  <p align="center">
  <img src="header.png" alt="Antagonistic dual-wavelength tomographic volumetric additive manufacturing">
  </p>
  
  <p align="center">
    <br />
    <a href="mailto:quinten.thijssen@ugent.be"><strong>Quinten Thijssen</strong></a>
    ·
    <a href="https://www.felixwechsler.science/"><strong>Felix Wechsler †</strong></a>
    ·
    <strong>Antonio J. Ortega †</strong></a>
    ·
    <strong>Joshua A. Carroll</strong></a>
    ·
    <strong>Christophe Moser</strong></a>
    ·
    <strong>Sandra Van Vlierberghe</strong></a>
    ·
    <a href="mailto:christopher.barnerkowollik@qut.edu.au"><strong>Christopher Barner-Kowollik</strong></a>
  </p>

  <p align="center">
    <a href='https://chemrxiv.org/doi/abs/10.26434/chemrxiv.15006647/v1'>
      <img src='https://img.shields.io/badge/Paper-PDF-red?style=flat-square' alt='Paper PDF'>
    </a>
  </p>
</p>

**a** School of Chemistry and Physics, Queensland University of Technology (QUT), 2 George Street, Brisbane, QLD 4000, Australia<br>
**b** Polymer Chemistry and Biomaterials Group, Centre of Macromolecular Chemistry, Department of Organic and Macromolecular Chemistry, Ghent University, Krijgslaan 291 S4, 9000, Belgium<br>
**c** Laboratory of Applied Photonics Devices, Institute of Electrical and Microengineering, Ecole Polytechnique Fédérale de Lausanne, Lausanne, Switzerland<br>
**d** Research Group in Design, Manufacturing and Materials (DM+M), School of Mechanical Engineering, Universidad Tecnológica de Panamá, Panama City 0819-07289, Panama<br>
**e** Institute of Functional Interfaces (IFG), Karlsruhe Institute of Technology (KIT), Herrmann-von-Helmholtz-Platz 1, 76344 Eggenstein-Leopoldshafen, Germany<br>

† Authors contributed equally

*Corresponding authors: quinten.thijssen@ugent.be, christopher.barnerkowollik@qut.edu.au*

## Antagonistic TVAM

Tomographic volumetric additive manufacturing can accumulate unwanted background dose outside the target as many projections overlap. Antagonistic dual-wavelength TVAM addresses this background by combining a patterned activation field with a separately optimised inhibition field that suppresses polymerisation where activation is not wanted.

This enables:

- suppression of tomographic background dose;
- confinement of individual features;
- definition of complete three-dimensional geometries through inhibition;
- extension of the accessible material window;
- mitigation of diffusion- and attenuation-driven loss of spatial fidelity.

## Principle

The effective field represents the activation remaining after spatially patterned inhibition:

$$
D_{\mathrm{eff}}(\mathbf{r}) =
\max \left[\alpha D_{405}(\mathbf{r}) - D_{365}(\mathbf{r}),\,0\right]
$$

Here, $D_{405}$ and $D_{365}$ are the reconstructed activation and inhibition fields. The 405 nm projection stack is fixed, while the non-negative 365 nm projection values are optimised. The factor $\alpha$ scales the activation field in the numerical objective.

The implementation builds on the open-source [Dr.TVAM](https://github.com/rgl-epfl/drtvam) framework and uses its optical forward model and differentiable optimisation workflow.

## Installation

Install the pinned Dr.TVAM revision and its dependencies:

```bash
python -m pip install -r requirements.txt
```

## Running the optimiser

Generate the fixed 405 nm activation patterns:

```bash
drtvam examples/background_suppression/activation_config.json --backend cuda -D output=output/background_suppression/activation
```

Optimise the non-negative 365 nm inhibition patterns:

```bash
python antagonistic_tvam/optimize_antagonistic.py --activation-config examples/background_suppression/activation_config.json --inhibition-config examples/background_suppression/inhibition_config.json --activation-patterns output/background_suppression/activation/patterns.npz --activation-scale 1.00 --backend cuda --output output/background_suppression/antagonistic
```

The first command generates the activation patterns. The second loads these patterns, holds the reconstructed activation field fixed and optimises the inhibition patterns. The resulting patterns and reconstructed fields are written to the ignored `output` directory.

## Examples

| Example | Purpose |
| --- | --- |
| `examples/background_suppression` | Suppression of activation between two cylindrical target volumes. |
| `examples/feature_confinement` | Confinement of line features within a broader rectangular activation field. |
| `examples/inhibition_encoded_geometry` | Definition of I-WP, diamond, Thinker and gyroid geometries through inhibition within cylindrical activation fields. |

## Simulations

- `simulations/chemical_window` illustrates how antagonistic inhibition can extend the accessible material window.
- `simulations/diffusion` illustrates how periodic inhibitor photogeneration can reduce diffusion-driven loss of spatial confinement.
- `simulations/attenuation` illustrates how patterned inhibition can suppress peripheral overexposure caused by optical attenuation.

Each script reproduces one fixed proof-of-principle scenario:

```bash
python simulations/chemical_window/run_simulation.py
python simulations/diffusion/run_simulation.py
python simulations/attenuation/run_simulation.py
```

## Reference

Q. Thijssen, F. Wechsler, A. J. Ortega, J. A. Carroll, C. Moser, S. Van Vlierberghe and C. Barner-Kowollik, “Antagonistic Dual-Wavelength Tomographic Volumetric Additive Manufacturing,” *ChemRxiv* (2026). [https://doi.org/10.26434/chemrxiv.15006647/v1](https://doi.org/10.26434/chemrxiv.15006647/v1)

B. Nicolet, F. Wechsler, J. Madrid-Wolff, C. Moser and W. Jakob, “Inverse Rendering for Tomographic Volumetric Additive Manufacturing,” *ACM Transactions on Graphics* **43**(6), Article 228 (2024).

## Licence

This repository is made available under the non-commercial Dr.TVAM licence provided in [`LICENSE`](LICENSE).
