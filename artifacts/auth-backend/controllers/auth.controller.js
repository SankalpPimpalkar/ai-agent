import authService from '../services/auth.service.js';
import jwt from 'jsonwebtoken';

export const register = (req, res) => {
  const { username, password } = req.body;
  if (!username || !password) return res.status(400).json({ message: 'Provide username and password' });

  try {
    authService.register(username, password);
    res.status(201).json({ message: 'User registered successfully' });
  } catch (error) {
    res.status(400).json({ message: error.message });
  }
};

export const login = (req, res) => {
  const { username, password } = req.body;
  try {
    const user = authService.login(username, password);
    const token = jwt.sign({ username: user.username }, process.env.JWT_SECRET || 'your_secret_key_here', { expiresIn: '1h' });
    res.json({ token });
  } catch (error) {
    res.status(401).json({ message: error.message });
  }
};
