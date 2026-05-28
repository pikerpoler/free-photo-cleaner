import React from 'react';
import {useColorScheme} from 'react-native';
import {NavigationContainer, DefaultTheme, DarkTheme} from '@react-navigation/native';
import {createBottomTabNavigator} from '@react-navigation/bottom-tabs';
import {PhotosScreen} from '../screens/PhotosScreen';
import {VideosScreen} from '../screens/VideosScreen';
import {SettingsScreen} from '../screens/SettingsScreen';

const Tab = createBottomTabNavigator();

export function AppNavigator() {
  const colorScheme = useColorScheme();

  return (
    <NavigationContainer
      theme={colorScheme === 'dark' ? DarkTheme : DefaultTheme}>
      <Tab.Navigator
        screenOptions={{
          headerShown: true,
          tabBarActiveTintColor: '#007AFF',
        }}>
        <Tab.Screen
          name="Photos"
          component={PhotosScreen}
          options={{
            tabBarLabel: 'Photos',
            tabBarIcon: () => null,
          }}
        />
        <Tab.Screen
          name="Videos"
          component={VideosScreen}
          options={{
            tabBarLabel: 'Videos',
            tabBarIcon: () => null,
          }}
        />
        <Tab.Screen
          name="Settings"
          component={SettingsScreen}
          options={{
            tabBarLabel: 'Settings',
            tabBarIcon: () => null,
          }}
        />
      </Tab.Navigator>
    </NavigationContainer>
  );
}
