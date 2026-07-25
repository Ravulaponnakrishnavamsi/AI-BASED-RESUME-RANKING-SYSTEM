"""
Authentication utilities for AI Recruitment Dashboard
Simple session-based authentication (replace with proper auth in production)
"""

import streamlit as st
import hashlib
import json
import os

USERS_FILE = "users.json"

def hash_password(password: str) -> str:
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    """Load users from JSON file"""
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    else:
        # Default demo user
        default_users = {
            "recruiter@ai.com": {
                "password": hash_password("demo123"),
                "name": "Demo Recruiter"
            }
        }
        save_users(default_users)
        return default_users

def save_users(users: dict):
    """Save users to JSON file"""
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)

def authenticate_user(username: str, password: str) -> bool:
    """Authenticate user credentials"""
    users = load_users()
    
    if username in users:
        return users[username]["password"] == hash_password(password)
    return False

def register_user(username: str, password: str, name: str) -> bool:
    """Register a new user"""
    users = load_users()
    
    if username in users:
        return False  # User already exists
    
    users[username] = {
        "password": hash_password(password),
        "name": name
    }
    save_users(users)
    return True

def is_authenticated() -> bool:
    """Check if user is authenticated"""
    return st.session_state.get("authenticated", False)

def get_current_user():
    """Get current logged-in user info"""
    return st.session_state.get("user_info", {})

def login_user(username: str, name: str):
    """Set user as logged in"""
    st.session_state.authenticated = True
    st.session_state.user_info = {
        "username": username,
        "name": name
    }

def logout_user():
    """Log out current user"""
    st.session_state.authenticated = False
    st.session_state.user_info = {}
