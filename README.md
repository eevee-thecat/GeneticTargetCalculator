# GeneticTargetCalculator

Some problems are easy to solve, like 5x + 3 = 10. Other problems are more difficult to solve.

Imagine you were given a puzzle: `_ * _ - _ / _ + _ * _ * _ = 67`, where you can only use the digits 1-9 in the blanks. Or, an even longer puzzle with more blanks - you get the picture.

You could brute force the solution, but for every blank the number of possibilities increases by a factor of 9. How could we solve this efficiently with Python?

## Genetic Algorithms

As humans, we'd try a few numbers in the blank and see how close we got to the solution, and then maybe do some tweaks from there. 

We can treat this trial-and-error process as an algorithm - generate some guesses, rank them by how "good" they were, and try to use the best guesses to make even better guesses.

In this case, we treat each blank of the equation as a "gene" in a simulated organism. The most successful organisms pair up with the other most successful organisms, and their genes mix, producing (hopefully) an even better new organism.

During the breeding process, there is a chance for mutations to occur - perhaps the set of organisms is "stuck" and needs some help