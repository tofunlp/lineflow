from .test_ag_news import AgNewsTestCase


class SogouNewsTestCase(AgNewsTestCase):

    def setUp(self):
        super(SogouNewsTestCase, self).setUp()
        self.name = self.names[1]
        self.size = self.sizes[1]
