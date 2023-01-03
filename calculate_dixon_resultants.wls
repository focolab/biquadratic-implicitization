#!/usr/bin/env wolframscript
triangleMatrix = Import["biquadratic_triangle_cayley_matrix_mathematica.txt", "List"] // ToExpression // First;
triangleResultant = Det[triangleMatrix];
Export["biquadratic_triangle_implicit.txt", triangleResultant]

quadrilateralMatrix = Import["biquadratic_quadrilateral_cayley_matrix_mathematica.txt", "List"] // ToExpression // First;
quadrilateralResultant = Det[quadrilateralMatrix];
Export["biquadratic_quadrilateral_implicit.txt", quadrilateralResultant]

(* count operations per https://mathematica.stackexchange.com/questions/210854/count-number-of-individual-operations *)
triangleLeaves = Level[triangleResultant, {-1}, Heads -> True];
Print[Tally[triangleLeaves]]
quadrilateralLeaves = Level[quadrilateralResultant, {-1}, Heads -> True];
Print[Tally[quadrilateralLeaves]]

transformedTriangleMatrix = Import["biquadratic_triangle_cayley_matrix_transformed_mathematica.txt", "List"] // ToExpression // First;
transformedTriangleResultant = Det[transformedTriangleMatrix];
Export["biquadratic_triangle_transformed_implicit.txt", transformedTriangleResultant]

transformedQuadrilateralMatrix = Import["biquadratic_quadrilateral_cayley_matrix_transformed_mathematica.txt", "List"] // ToExpression // First;
transformedQuadrilateralResultant = Det[transformedQuadrilateralMatrix];
Export["biquadratic_quadrilateral_transformed_implicit.txt", transformedQuadrilateralResultant]

transformedTriangleLeaves = Level[transformedTriangleResultant, {-1}, Heads -> True];
Print[Tally[transformedTriangleLeaves]]
transformedQuadrilateralLeaves = Level[transformedQuadrilateralResultant, {-1}, Heads -> True];
Print[Tally[transformedQuadrilateralLeaves]]