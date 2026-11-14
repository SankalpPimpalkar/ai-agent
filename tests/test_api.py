import unittest
from fastapi.testclient import TestClient
from src.agent.main import build_app
from src.agent.api.schemas import Query


class TestApiEndpoints(unittest.TestCase):
    def setUp(self):
        self.app = build_app()

    def test_routes_registered(self):
        routes = [route.path for route in self.app.routes if hasattr(route, "path")]
        self.assertIn("/api/stream", routes)
        self.assertIn("/api/chat", routes)
        self.assertIn("/api/tool", routes)

    def test_app_state_configured(self):
        self.assertIsNotNone(self.app.state.settings)
        self.assertIsNotNone(self.app.state.client)


if __name__ == "__main__":
    unittest.main()
