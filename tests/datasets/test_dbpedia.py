from .test_ag_news import AgNewsTestCase


class DbpediaTestCase(AgNewsTestCase):

    def setUp(self):
        super(DbpediaTestCase, self).setUp()
        self.name = self.names[2]
        self.size = self.sizes[2]
