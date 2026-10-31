import bcrypt from 'bcryptjs';
import { User } from './user.model.js';

export class AuthService {
  static async register(username, password) {
    const existingUser = await User.findOne({ username });
    if (existingUser) throw new Error('User already exists');

    const hashedPassword = await bcrypt.hash(password, 10);
    await User.create({ username, password: hashedPassword });
    return { message: 'User registered successfully' };
  }

  static async login(username, password) {
    const user = await User.findOne({ username });
    if (!user || !(await bcrypt.compare(password, user.password))) {
      throw new Error('Invalid credentials');
    }
    return user;
  }
}
