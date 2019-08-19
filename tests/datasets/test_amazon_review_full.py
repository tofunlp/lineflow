from .test_ag_news import AgNewsTestCase


class AmazonReviewFullTestCase(AgNewsTestCase):

    def setUp(self):
        super(AmazonReviewFullTestCase, self).setUp()
        self.name = self.names[7]
        self.size = self.sizes[7]
