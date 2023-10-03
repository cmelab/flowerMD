from hoomd_organics.library import Graphene


class TestSurfaces:
    def test_create_graphene_surface(self):
        surface = Graphene(
            x_repeat=2,
            y_repeat=2,
            n_layers=3,
            periodicity=(False, False, False),
        )
        assert surface.n_particles == 4 * 2 * 2 * 3
        assert surface.n_bonds == 54

    def test_graphene_with_periodicity(self):
        surface = Graphene(
            x_repeat=2, y_repeat=2, n_layers=3, periodicity=(True, True, False)
        )
        assert surface.n_particles == 4 * 2 * 2 * 3
        assert surface.periodicity == (True, True, False)
        assert surface.n_bonds == 72
