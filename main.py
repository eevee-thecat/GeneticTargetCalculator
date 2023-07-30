from __future__ import annotations

import math
import string

import numexpr as ne
import itertools

from dataclasses import dataclass
import random

from numpy import ndarray

RANDOM_MUTATION_CHANCE = 0.1  # Every time we breed, there is a RANDOM_MUTATION_CHANCE to change the value of a gene
RANDOM_MUTATION_MAGNITUDE = 5  # If the RANDOM_MUTATION_CHANCE is hit, we change the value of a gene up to this magnitude

MIN_GENE_VALUE = 1  # The value of a gene cannot go below this
MAX_GENE_VALUE = 99  # The value of a gene cannot go above this

LEADERS = 5  # The top LEADERS genomes pass by unchanged to the next round
SURVIVORS = 15  # Keep SURVIVORS genomes out of all the entire pool after breeding + random spawning

MAX_ROUNDS = 200  # Give up after MAX_ROUNDS of evolution


def alternate(*iters):
    for row in itertools.zip_longest(*iters):
        for i in row:
            yield i


@dataclass
class Gene:
    value: int
    min_value: int
    max_value: int

    def __str__(self) -> str:
        return str(self.value)

    def mutate(self):
        new_value = self.value + random.randint(
            -RANDOM_MUTATION_MAGNITUDE, RANDOM_MUTATION_MAGNITUDE
        )
        new_value = min(self.max_value, new_value)
        new_value = max(self.min_value, new_value)
        return Gene(value=new_value, min_value=self.min_value, max_value=self.max_value)

    def breed(self, other: Gene):
        new_value = round(self.value / 2 + other.value / 2)
        new_value = min(self.max_value, new_value)
        new_value = max(self.min_value, new_value)
        result = Gene(
            value=new_value, min_value=self.min_value, max_value=self.max_value
        )
        if random.random() < RANDOM_MUTATION_CHANCE:
            return result.mutate()
        else:
            return result


@dataclass
class Genome:
    genes: list[Gene]

    def breed(self, other: Genome) -> Genome:
        new_genes = []
        for left, right in zip(self.genes, other.genes):
            new_genes.append(left.breed(right))

        return Genome(genes=new_genes)

    def __lt__(self, other):
        return True


@dataclass
class EquationObjective:
    equation: list[str]
    expected: float

    def get_value(self, genome: Genome) -> ndarray:
        full_equation = list(
            alternate((str(gene) for gene in genome.genes), self.equation)
        )[:-1]
        full_equation = "".join(full_equation)
        actual = ne.evaluate(full_equation)

        return actual

    def evaluate(self, genome: Genome) -> float:
        full_equation = list(
            alternate((str(gene) for gene in genome.genes), self.equation)
        )[:-1]
        full_equation = "".join(full_equation)
        actual = ne.evaluate(full_equation)

        return abs(self.expected - actual)

    def to_str(self, genome: Genome) -> str:
        full_equation = list(
            alternate((str(gene) for gene in genome.genes), self.equation)
        )[:-1]
        full_equation = " ".join(full_equation)

        return full_equation


def random_genome(num_genes: int) -> Genome:
    genes = [
        Gene(
            value=random.randint(MIN_GENE_VALUE, MAX_GENE_VALUE),
            min_value=MIN_GENE_VALUE,
            max_value=MAX_GENE_VALUE,
        )
        for _ in range(num_genes)
    ]
    genome = Genome(genes=genes)

    return genome


def evolve(objective, initial_genomes, num_genes) -> tuple[list[Genome], bool]:
    genomes_and_fitness: list[tuple[float, Genome]] = [
        (objective.evaluate(genome), genome) for genome in initial_genomes
    ]

    genomes_and_fitness.sort()

    # Save the leaders
    new_genomes = initial_genomes[:LEADERS]

    for _ in range(4):
        new_genomes.append(random_genome(num_genes))
    # Breed
    for left_index in range(0, len(initial_genomes)):
        for right_index in range(left_index + 1, len(initial_genomes)):
            new_genomes.append(
                initial_genomes[left_index].breed(initial_genomes[right_index])
            )

    new_genomes_and_fitness: list[tuple[float, Genome]] = [
        (objective.evaluate(genome), genome) for genome in new_genomes
    ]
    new_genomes_and_fitness.sort()

    unique_fitnesses = list(set(pair[0] for pair in new_genomes_and_fitness))
    unique_fitnesses.sort()

    print(
        round(unique_fitnesses[0], 2),
        f"({objective.to_str(new_genomes_and_fitness[0][1])})",
    )
    print(
        f"Next 3 fitnesses: ", *(round(fitness, 2) for fitness in unique_fitnesses[1:4])
    )

    # if objective.expected.is_integer():
    #     # Can use exact equality
    #     if unique_fitnesses == 0:
    #         return new_genomes, True
    # else:
    # Use isclose to check float "equality"
    if math.isclose(unique_fitnesses[0], 0, abs_tol=0.05):
        return new_genomes, True

    # Cull all but SURVIVORS
    new_genomes = [pair[1] for pair in new_genomes_and_fitness[:SURVIVORS]]

    return new_genomes, False


def get_operators_from_input() -> tuple[list[str], float]:
    print("Please input an equation that you'd like to solve for")
    print(
        """Some examples:
a + b * c - d / e = 32.5
a * b - c + d + e / f - g * h + i - j = 64
"""
    )
    print(
        "Only the characters '+' '*' '-' '/' and the final '= TARGET_NUMBER' will be considered."
    )
    print("Target values between 1 and ~600 work best.")
    print(
        "Since this is a stochastic process and some equations are harder to solve for than others (or are impossible to solve), not every attempt will work. Play around with the UPPER_CASE constants and the equation!\n"
    )
    raw_equation = input("")
    operators = []

    for char in raw_equation:
        if char in {"+", "*", "-", "/"}:
            operators.append(char)

    target_value = float(raw_equation[raw_equation.find("=") + 1 :].strip())

    return operators, target_value


def main():
    choice = int(
        input(
            "Please enter 0 to solve a random equation, or 1 to input your own equation to solve:\n"
        )
    )
    if choice == 0:
        operators = random.choices("+-*/", k=random.randint(4, 12))
        target_value = round(random.uniform(1, 300), 2)
        print(
            "Since this is a stochastic process and some equations are harder to solve for than others (or are impossible to solve), not every attempt will work. Play around with the UPPER_CASE constants and the equation!\n"
        )
        print(
            f"Trying to solve for: {' '.join(alternate(string.ascii_lowercase[: len(operators)], operators))} = {target_value}"
        )
        input("Enter anything to continue... ")
    elif choice == 1:
        operators, target_value = get_operators_from_input()
    else:
        print("Error: invalid choice")
        exit(1)
    objective = EquationObjective(
        equation=operators,
        expected=target_value,
    )
    num_genes = len(operators) + 1
    genomes: list[Genome] = [random_genome(num_genes) for _ in range(SURVIVORS)]

    for step in range(1, MAX_ROUNDS + 1):
        print(f"Round #{step} \tBest fitness: ", end="")
        new_genomes, target_found = evolve(objective, genomes, num_genes)
        genomes = new_genomes
        print()
        if target_found:
            print(
                f"Solution found: {objective.to_str(genomes[0])} ~= {objective.expected}"
            )
            break
    else:
        print(f"Solution not found in {MAX_ROUNDS} rounds")
        print(
            f"Target: {objective.expected}\tBest answer: {round(float(objective.get_value(genomes[0])), 2)}\tGenome: {objective.to_str(genomes[0])}"
        )


if __name__ == "__main__":
    # random.seed(0)
    main()
