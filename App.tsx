import React, {useEffect} from 'react';
import {GestureHandlerRootView} from 'react-native-gesture-handler';
import {StyleSheet} from 'react-native';
import {AppNavigator} from './src/navigation/AppNavigator';
import {getDatabase} from './src/services/database';

function App(): React.JSX.Element {
  useEffect(() => {
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
