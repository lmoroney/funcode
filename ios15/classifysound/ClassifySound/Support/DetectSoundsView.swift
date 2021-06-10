/*
See LICENSE folder for this sample’s licensing information.

Abstract:
A view for visualizing the confidence with which the system is detecting sounds
from the audio input.
*/

import Foundation
import SwiftUI

///  Provides a visualization the app uses when detecting sounds.
struct DetectSoundsView: View {
    var emojiDictionary:Dictionary<String, String> = EmojiDictionaryHelper.init().emoji_dictionary
    /// The runtime state that contains information about the strength of the detected sounds.
    @ObservedObject var state: AppState

    /// The configuration that dictates aspects of sound classification, as well as aspects of the visualization.
    @Binding var config: AppConfiguration

    /// An action to perform when the user requests to edit the app's configuration.
    let configureAction: () -> Void



    /// Generates a grid of confidence meters that indicate sounds the app detects.
    ///
    /// - Parameter detections: A list of sounds that contain their detection states to render. The
    ///   states indicate whether the sounds appear visible, and if so, the level of confidence to render.
    ///
    /// - Returns: A view that contains a grid of confidence meters, indicating which sounds the app
    ///   detects and how strongly.
    static func generateDetectionsGrid(_ detections: [(SoundIdentifier, DetectionState)], dictionary: Dictionary<String,String>) -> some View {
        return ScrollView {
            ForEach(detections, id: \.0.labelName) {
                if($0.1.currentConfidence>0.3){
                    Label(dictionary[$0.0.labelName]!, systemImage: "").font(.system(size:120))
                }
            }
        }
    }

    var body: some View {
        VStack {
            ZStack {
                VStack {
                    Text("Detecting Sounds").font(.title).padding()
                    DetectSoundsView.generateDetectionsGrid(state.detectionStates, dictionary: emojiDictionary)
                }.disabled(!state.soundDetectionIsRunning)
            }

            Button(action: configureAction) {
                Text("Edit Configuration")
            }.padding()
        }
    }
}
