"""Run every suite: inference/cohort checks, then the comparator checks."""
import sim.tests
import model.tests

if __name__ == "__main__":
    sim.tests.main()
    print()
    model.tests.main()
