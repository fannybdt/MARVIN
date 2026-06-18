# Available Data
Both datasets are ready-to-use and preprocessed using the *asinh* transform, followed by standardization.
## AML Dataset
Originates from [Levine J.H. et al, Data-Driven Phenotypic Dissection of AML Reveals Progenitor-like Cells that Correlate with Prognosis (2015)
](https://www.sciencedirect.com/science/article/pii/S0092867415006376?via%3Dihub) as *Benchmark Dataset 2*, it consists of samples from two healthy adult donors (note: despite the name "AML" referring to the original study, these particular donors are healthy). The data were acquired using mass cytometry (CyTOF), measuring 32 surface markers on bone marrow mononuclear cells (BMMCs). 

The samples were manually gated into 14 distinct immune cell types: the dataset comprises a
total of 104,184 fully annotated cells.

## BMMC Dataset
The second dataset comes from [Bendall et al., Single-Cell Mass Cytometry of Differential Immune and Drug Responses Across a Human Hematopoietic Continuum (2011) ](https://doi.org/10.1126/science.1198704), consisting of 61,725 bone marrow
mononuclear cells from a single individual.
This dataset was also obtained via mass cytometry, with 13 markers measured to delineate 19
distinct cell populations.