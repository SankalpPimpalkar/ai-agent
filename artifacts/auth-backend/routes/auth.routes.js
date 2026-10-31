import express from 'express';
import authRoutes from './auth.routes.js';

const router = express.Router();

router.post('/register', authRoutes.register);
router.post('/login', authRoutes.login);

export default router;
