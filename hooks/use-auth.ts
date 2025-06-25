// Authentication hook using Zustand for state management
// Handles login, signup, logout, and authentication state

'use client';

import { create } from 'zustand';
import { User } from '@/types';
import { authApi, tokenManager } from '@/lib/api-client';

interface AuthState {
  user: User | null;
  isLoading: boolean;
  isAuthenticated: boolean;
  login: (email: string, password: string, rememberMe?: boolean) => Promise<void>;
  signup: (name: string, email: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
  initialize: () => void;
}

export const useAuth = create<AuthState>((set, get) => ({
  user: null,
  isLoading: false,
  isAuthenticated: false,

  login: async (email: string, password: string, rememberMe = false) => {
    set({ isLoading: true });
    try {
      const response = await authApi.login({ email, password, rememberMe });
      
      // Store token and user data
      tokenManager.setToken(response.token);
      tokenManager.setUser(response.user);
      
      set({ 
        user: response.user, 
        isAuthenticated: true, 
        isLoading: false 
      });
    } catch (error) {
      set({ isLoading: false });
      throw error;
    }
  },

  signup: async (name: string, email: string, password: string) => {
    set({ isLoading: true });
    try {
      const response = await authApi.signup({ name, email, password });
      
      // Store token and user data
      tokenManager.setToken(response.token);
      tokenManager.setUser(response.user);
      
      set({ 
        user: response.user, 
        isAuthenticated: true, 
        isLoading: false 
      });
    } catch (error) {
      set({ isLoading: false });
      throw error;
    }
  },

  logout: async () => {
    await authApi.logout();
    set({ user: null, isAuthenticated: false });
  },

  initialize: () => {
    // Check for existing authentication on app load
    const user = tokenManager.getUser();
    const token = tokenManager.getToken();
    const isAuthenticated = !!(user && token);
    
    set({ user, isAuthenticated });
  }
}));