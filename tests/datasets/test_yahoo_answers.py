from .test_ag_news import AgNewsTestCase


class YahooAnswersTestCase(AgNewsTestCase):

    def setUp(self):
        super(YahooAnswersTestCase, self).setUp()
        self.name = self.names[5]
        self.size = self.sizes[5]
