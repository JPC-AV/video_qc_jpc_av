#!/bin/sh
# Get version from pyproject.toml
VERSION=$(grep -m 1 "version" pyproject.toml | grep -o '"[^"]*"' | sed 's/"//g')
echo "Building AV-Spex version $VERSION"
# Create a folder (named dmg) to prepare our DMG in
mkdir -p dist/dmg
# Empty the dmg folder
rm -r dist/dmg/*
# Copy the app bundle to the dmg folder
cp -R "dist/AV-Spex.app" dist/dmg
# Make sure permissions are correct
chmod -R 755 dist/dmg/AV-Spex.app

# Create entitlements file if it doesn't exist
cat > entitlements.plist << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>com.apple.security.cs.allow-jit</key>
    <true/>
    <key>com.apple.security.cs.allow-unsigned-executable-memory</key>
    <true/>
    <key>com.apple.security.cs.disable-library-validation</key>
    <true/>
    <key>com.apple.security.cs.allow-dyld-environment-variables</key>
    <true/>
    <key>com.apple.security.automation.apple-events</key>
    <true/>
</dict>
</plist>
EOF

# Sign each framework/library/plugin individually first
echo "Signing frameworks and libraries..."
find dist/dmg/AV-Spex.app/Contents/Frameworks -type f -name "*.so" -o -name "*.dylib" | while read file; do
  codesign --force --timestamp --options runtime --sign "Developer ID Application: Eddy Colloton (4A8B3AQ4VX)" "$file"
done

# Sign Python executables
echo "Signing Python executables..."
find dist/dmg/AV-Spex.app/Contents -type f -name "python*" | while read file; do
  codesign --force --timestamp --options runtime --sign "Developer ID Application: Eddy Colloton (4A8B3AQ4VX)" "$file"
done

# Sign any other executables in MacOS folder
echo "Signing other executables..."
find dist/dmg/AV-Spex.app/Contents/MacOS -type f -perm +111 | while read file; do
  codesign --force --timestamp --options runtime --entitlements entitlements.plist --sign "Developer ID Application: Eddy Colloton (4A8B3AQ4VX)" "$file"
done

# Sign the main app bundle with entitlements
echo "Signing main application bundle..."
codesign --force --deep --timestamp --options runtime --entitlements entitlements.plist --sign "Developer ID Application: Eddy Colloton (4A8B3AQ4VX)" dist/dmg/AV-Spex.app

# If the DMG already exists, delete it
test -f "dist/AV-Spex-$VERSION.dmg" && rm "dist/AV-Spex-$VERSION.dmg"

# Create the DMG with version number
create-dmg \
  --volname "AV-Spex $VERSION" \
  --volicon "av_spex_the_logo.icns" \
  --window-pos 200 120 \
  --window-size 600 300 \
  --icon-size 100 \
  --icon "AV-Spex.app" 175 120 \
  --hide-extension "AV-Spex.app" \
  --app-drop-link 425 120 \
  --no-internet-enable \
  "dist/AV-Spex-$VERSION.dmg" \
  "dist/dmg"

# Sign the DMG itself
codesign --force --timestamp --sign "Developer ID Application: Eddy Colloton (4A8B3AQ4VX)" "dist/AV-Spex-$VERSION.dmg"

# Notarize the DMG
echo "Submitting DMG for notarization..."
xcrun notarytool submit "dist/AV-Spex-$VERSION.dmg" --keychain-profile "DEV_CERT_PW" --wait --timeout 1800

# After notarization completes, staple the ticket to the DMG
echo "Stapling ticket to DMG..."
xcrun stapler staple "dist/AV-Spex-$VERSION.dmg"

echo "Build and notarization process complete for AV-Spex version $VERSION!"