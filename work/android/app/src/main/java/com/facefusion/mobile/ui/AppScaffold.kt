package com.facefusion.mobile.ui

import androidx.compose.foundation.layout.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Face
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

enum class Screen { Swap, Settings }

/**
 * The frame around both screens: wordmark above, two destinations below.
 *
 * Two destinations is not enough to justify a navigation library -- and adding one would
 * mean resolving a dependency this build cannot be relied on to fetch. A plain enum plus
 * Material3's own NavigationBar is the whole navigation system.
 */
@Composable
fun AppScaffold(
    screen: Screen,
    onScreen: (Screen) -> Unit,
    content: @Composable (PaddingValues) -> Unit,
) {
    Scaffold(
        topBar = {
            Row(
                Modifier
                    .fillMaxWidth()
                    // targetSdk 35 makes the window edge-to-edge on Android 15, and
                    // Scaffold insets its CONTENT but not its topBar -- so without this the
                    // wordmark sits under the status bar and behind the camera cutout.
                    .statusBarsPadding()
                    .padding(start = 16.dp, end = 16.dp, top = 8.dp, bottom = 18.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(10.dp),
            ) {
                AppMark()
                // The tier is not shown here any more; Settings reports it alongside
                // the arch, VTCM and fp16 verdict, which is where it means something.
                Wordmark(Modifier.weight(1f))
            }
        },
        bottomBar = {
            NavigationBar {
                NavigationBarItem(
                    selected = screen == Screen.Swap,
                    onClick = { onScreen(Screen.Swap) },
                    icon = { Icon(Icons.Default.Face, "Swap") },
                    label = { Text("Swap") },
                )
                NavigationBarItem(
                    selected = screen == Screen.Settings,
                    onClick = { onScreen(Screen.Settings) },
                    icon = { Icon(Icons.Default.Settings, "Settings") },
                    label = { Text("Settings") },
                )
            }
        },
        content = content,
    )
}
