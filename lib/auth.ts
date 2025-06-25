import Cookies from 'js-cookie';

export interface User {
  id: string;
  name: string;
  email: string;
  role: 'doctor' | 'admin';
}

export interface LoginCredentials {
  email: string;
  password: string;
  rememberMe?: boolean;
}

export interface SignupData {
  name: string;
  email: string;
  password: string;
}

// Mock API calls - replace with actual FastAPI endpoints
export const authApi = {
  async login(credentials: LoginCredentials): Promise<{ user: User; token: string }> {
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    // Mock successful login
    if (credentials.email === 'doctor@sanamed.com' && credentials.password === 'password') {
      const user: User = {
        id: '1',
        name: 'Dr. Sarah Johnson',
        email: credentials.email,
        role: 'doctor'
      };
      const token = 'mock-jwt-token-' + Date.now();
      
      return { user, token };
    }
    
    throw new Error('Invalid credentials');
  },

  async signup(data: SignupData): Promise<{ user: User; token: string }> {
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    const user: User = {
      id: Date.now().toString(),
      name: data.name,
      email: data.email,
      role: 'doctor'
    };
    const token = 'mock-jwt-token-' + Date.now();
    
    return { user, token };
  },

  async logout(): Promise<void> {
    // Clear token
    Cookies.remove('auth-token');
    Cookies.remove('user-data');
  },

  getCurrentUser(): User | null {
    try {
      const userData = Cookies.get('user-data');
      return userData ? JSON.parse(userData) : null;
    } catch {
      return null;
    }
  },

  getToken(): string | null {
    return Cookies.get('auth-token') || null;
  },

  isAuthenticated(): boolean {
    return !!this.getToken();
  }
};

export const setAuthCookies = (user: User, token: string, rememberMe = false) => {
  const expires = rememberMe ? 30 : 1; // 30 days or 1 day
  
  Cookies.set('auth-token', token, { expires });
  Cookies.set('user-data', JSON.stringify(user), { expires });
};