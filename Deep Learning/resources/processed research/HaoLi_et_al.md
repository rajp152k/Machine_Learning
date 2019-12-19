# VISUALISING THE LOSS LANDSCAPE OF NEURAL NETS

 - ways to visualize the loss contour as a two dimnensional surface
	- generic idea: for loss visualization: use PCA to reduce to 2 dimensions and plot from a heuristically chosen "center" ( the parameter matrix ) and expore in different directions ( an affine combination ).
	- PCA on the Hessian ( use eigenvectors with largest corresponding eigenvalue)
- application: explains effectiveness of skip connections: via concatenation(dense net) and scaled addition ( resnet ) : helps achieve a flatter contour : eliminating non-convexitites.
n 
