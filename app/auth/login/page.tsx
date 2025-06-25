'use client';

import { useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/hooks/use-auth';
import { AuthGuard } from '@/components/auth/auth-guard';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Checkbox } from '@/components/ui/checkbox';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { toast } from 'sonner';
import { Stethoscope, Mail, Lock, Facebook, Chrome, Apple, AlertCircle } from 'lucide-react';

export default function LoginPage() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [rememberMe, setRememberMe] = useState(false);
  const [errors, setErrors] = useState<{email?: string; password?: string}>({});
  const { login, isLoading } = useAuth();
  const router = useRouter();

  const validateForm = () => {
    const newErrors: {email?: string; password?: string} = {};
    
    if (!email) {
      newErrors.email = 'Email is required';
    } else if (!/\S+@\S+\.\S+/.test(email)) {
      newErrors.email = 'Please enter a valid email address';
    }
    
    if (!password) {
      newErrors.password = 'Password is required';
    } else if (password.length < 6) {
      newErrors.password = 'Password must be at least 6 characters';
    }
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!validateForm()) {
      return;
    }

    try {
      await login(email, password, rememberMe);
      toast.success('Welcome back!');
      router.push('/dashboard');
    } catch (error) {
      toast.error(error instanceof Error ? error.message : 'Login failed');
    }
  };

  return (
    <AuthGuard requireAuth={false}>
      <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-teal-50 flex items-center justify-center p-4 medical-pattern">
        <div className="w-full max-w-md">
          {/* Logo */}
          <div className="text-center mb-8">
            <div className="w-16 h-16 medical-gradient rounded-2xl flex items-center justify-center mx-auto mb-4 shadow-lg">
              <Stethoscope className="w-8 h-8 text-white" />
            </div>
            <h1 className="text-3xl font-bold text-gray-900">Welcome Back!</h1>
            <p className="text-gray-600 mt-2">Sign in to your SanaMed account</p>
          </div>

          <Card className="shadow-2xl border-0 bg-white/95 backdrop-blur-sm">
            <CardHeader className="space-y-1 pb-6">
              <CardTitle className="text-2xl text-center">Sign in</CardTitle>
              <CardDescription className="text-center">
                Enter your credentials to access your account
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Social Login Options */}
              <div className="space-y-3">
                <p className="text-center text-sm text-gray-600">Sign in with</p>
                <div className="grid grid-cols-3 gap-3">
                  <Button variant="outline" className="w-full" disabled>
                    <Facebook className="h-4 w-4" />
                  </Button>
                  <Button variant="outline" className="w-full" disabled>
                    <Chrome className="h-4 w-4" />
                  </Button>
                  <Button variant="outline" className="w-full" disabled>
                    <Apple className="h-4 w-4" />
                  </Button>
                </div>
                
                <div className="relative">
                  <div className="absolute inset-0 flex items-center">
                    <span className="w-full border-t" />
                  </div>
                  <div className="relative flex justify-center text-xs uppercase">
                    <span className="bg-white px-2 text-muted-foreground">or</span>
                  </div>
                </div>
              </div>

              <form onSubmit={handleSubmit} className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="email">Your email address</Label>
                  <div className="relative">
                    <Mail className="absolute left-3 top-3 h-4 w-4 text-gray-400" />
                    <Input
                      id="email"
                      type="email"
                      placeholder="doctor@sanamed.com"
                      value={email}
                      onChange={(e) => {
                        setEmail(e.target.value);
                        if (errors.email) setErrors(prev => ({...prev, email: undefined}));
                      }}
                      className={`pl-10 ${errors.email ? 'border-red-500 focus:border-red-500' : ''}`}
                      required
                    />
                  </div>
                  {errors.email && (
                    <div className="flex items-center space-x-1 text-red-600 text-sm">
                      <AlertCircle className="w-4 h-4" />
                      <span>{errors.email}</span>
                    </div>
                  )}
                </div>

                <div className="space-y-2">
                  <Label htmlFor="password">Your password</Label>
                  <div className="relative">
                    <Lock className="absolute left-3 top-3 h-4 w-4 text-gray-400" />
                    <Input
                      id="password"
                      type="password"
                      placeholder="Enter your password"
                      value={password}
                      onChange={(e) => {
                        setPassword(e.target.value);
                        if (errors.password) setErrors(prev => ({...prev, password: undefined}));
                      }}
                      className={`pl-10 ${errors.password ? 'border-red-500 focus:border-red-500' : ''}`}
                      required
                    />
                  </div>
                  {errors.password && (
                    <div className="flex items-center space-x-1 text-red-600 text-sm">
                      <AlertCircle className="w-4 h-4" />
                      <span>{errors.password}</span>
                    </div>
                  )}
                </div>

                <div className="flex items-center space-x-2">
                  <Checkbox
                    id="remember"
                    checked={rememberMe}
                    onCheckedChange={(checked) => setRememberMe(checked as boolean)}
                  />
                  <Label
                    htmlFor="remember"
                    className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
                  >
                    Remember me
                  </Label>
                </div>

                <Button
                  type="submit"
                  className="w-full medical-gradient text-white py-3 text-lg"
                  disabled={isLoading}
                >
                  {isLoading ? (
                    <div className="flex items-center space-x-2">
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                      <span>Signing In...</span>
                    </div>
                  ) : (
                    'SIGN IN'
                  )}
                </Button>
              </form>

              {/* Demo Credentials */}
              <div className="bg-blue-50 p-4 rounded-lg text-sm border border-blue-200">
                <p className="text-blue-800 font-medium mb-2">Demo Credentials:</p>
                <div className="space-y-1 text-blue-700">
                  <p><strong>Email:</strong> doctor@sanamed.com</p>
                  <p><strong>Password:</strong> password</p>
                </div>
                <p className="text-blue-600 text-xs mt-2">
                  Use these credentials to explore the application
                </p>
              </div>

              <div className="text-center space-y-3">
                <Link
                  href="/auth/forgot-password"
                  className="text-sm text-primary hover:underline"
                >
                  Forgot your password?
                </Link>
                <p className="text-sm text-gray-600">
                  Don't have an account?{' '}
                  <Link href="/auth/signup" className="text-primary hover:underline font-medium">
                    Sign up
                  </Link>
                </p>
              </div>
            </CardContent>
          </Card>

          {/* Additional Info */}
          <div className="text-center mt-6">
            <p className="text-xs text-gray-500">
              By signing in, you agree to our{' '}
              <Link href="/terms" className="underline hover:text-gray-700">Terms of Service</Link>
              {' '}and{' '}
              <Link href="/privacy" className="underline hover:text-gray-700">Privacy Policy</Link>
            </p>
          </div>
        </div>
      </div>
    </AuthGuard>
  );
}