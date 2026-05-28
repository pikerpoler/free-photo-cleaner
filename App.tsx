import React, {useEffect} from 'react';
import {GestureHandlerRootView} from 'react-native-gesture-handler';
import {StyleSheet} from 'react-native';
import {AppNavigator} from './src/navigation/AppNavigator';
import {useSettingsStore} from './src/stores/settingsStore';
import {getDatabase} from './src/services/database';

function App(): React.JSX.Element {
  useEffect(() => {
    useSettingsStore.getState().loadSettings();
    getDatabase();
  }, []);

  return (
    <GestureHandlerRootView style={styles.root}>
      <AppNavigator />
    </GestureHandlerRootView>
  );
}

const styles = StyleSheet.create({
  root: {
    flex: 1,
  },
});

export default App;
