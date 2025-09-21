import streamlit as st
import pandas as pd
import pickle
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import time
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="Traffic Prediction Demo",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the model
@st.cache_resource
def load_model():
    """Load the XGBoost model"""
    try:
        with open('D:/eco_project/traffic_model_1h.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def create_sample_data_from_timestamp(timestamp_obj):
    """Create sample input based on a timestamp from the dataset"""
    # Handle both string and Timestamp objects
    if isinstance(timestamp_obj, str):
        dt = datetime.strptime(timestamp_obj, '%d/%m/%Y %H:%M')
    else:
        # timestamp_obj is already a pandas Timestamp, use it directly
        dt = timestamp_obj
    
    return {
        'month': dt.month,
        'day': dt.day,
        'minute': dt.minute,
        'hour': dt.hour,
        'day_of_week': dt.strftime('%A')
    }

# Load the training dataset (has target_next_1h for validation)
@st.cache_data
def load_data():
    """Load the traffic training dataset"""
    try:
        df = pd.read_csv('D:/eco_project/traffic_dataset1.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d/%m/%Y %H:%M')
        
        # Encode day_of_week to numeric for consistency
        df['day_of_week_encoded'] = df['day_of_week'].map({
            'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
            'Friday': 4, 'Saturday': 5, 'Sunday': 6
        })
        
        return df
    except Exception as e:
        st.error(f"Error loading training data: {e}")
        return None

# Load current data for prediction (no target_next_1h) - NO CACHE for real-time updates
def load_current_data():
    """Load the current data for prediction - refreshes every time to get latest data"""
    try:
        csv_path = 'D:/eco_project/Dashboard_update/ai-detect-traffic/bmatraffic_yolo_pipeline/src/data/sathon12_footbridge_id996_5min.csv'
        
        # Check if file exists
        if not os.path.exists(csv_path):
            st.error(f"CSV file not found: {csv_path}")
            return None
            
        # Get file modification time for freshness check
        file_mod_time = datetime.fromtimestamp(os.path.getmtime(csv_path))
        
        # Load CSV without any caching
        df = pd.read_csv(csv_path)
        
        if df.empty:
            st.warning("CSV file is empty")
            return None
            
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d/%m/%Y %H:%M')
        
        # Encode day_of_week to numeric for consistency
        df['day_of_week_encoded'] = df['day_of_week'].map({
            'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
            'Friday': 4, 'Saturday': 5, 'Sunday': 6
        })
        
        # Add file metadata for debugging
        df.attrs['file_mod_time'] = file_mod_time
        df.attrs['records_count'] = len(df)
        df.attrs['load_time'] = datetime.now()
        
        return df
    except Exception as e:
        st.error(f"Error loading current data: {e}")
        return None

# Load CCTV images
def load_cctv_images():
    """Load CCTV images - raw and detected with force refresh"""
    images = {}
    
    # Raw image path
    raw_image_path = 'D:/eco_project/Dashboard_update/ai-detect-traffic/bmatraffic_yolo_pipeline/src/data/snapshots/sathon12_footbridge_id996.jpg.raw.jpg'
    # Detected image path
    detected_image_path = 'D:/eco_project/Dashboard_update/ai-detect-traffic/bmatraffic_yolo_pipeline/src/data/snapshots/sathon12_footbridge_id996.jpg'

    try:
        # Load raw image
        if os.path.exists(raw_image_path):
            # Add timestamp to force reload from disk
            import time
            current_time = time.time()
            images['raw'] = Image.open(raw_image_path)
            images['raw_timestamp'] = datetime.fromtimestamp(os.path.getmtime(raw_image_path))
            images['raw_load_time'] = current_time
        else:
            images['raw'] = None
            images['raw_timestamp'] = None
            images['raw_load_time'] = None
            
        # Load detected image
        if os.path.exists(detected_image_path):
            current_time = time.time()
            images['detected'] = Image.open(detected_image_path)
            images['detected_timestamp'] = datetime.fromtimestamp(os.path.getmtime(detected_image_path))
            images['detected_load_time'] = current_time
        else:
            images['detected'] = None
            images['detected_timestamp'] = None
            images['detected_load_time'] = None
            
    except Exception as e:
        st.error(f"Error loading CCTV images: {e}")
        images['raw'] = None
        images['detected'] = None
        images['raw_timestamp'] = None
        images['detected_timestamp'] = None
        images['raw_load_time'] = None
        images['detected_load_time'] = None
    
    return images

def prepare_model_input(vehicle_count, lag_1, lag_2, lag_3, day_of_week, hour, month, day, minute):
    """Prepare input data in the format expected by the model"""
    # Model uses these 6 features: ['vehicle_count', 'lag_1', 'lag_2', 'lag_3', 'hour', 'day_of_week_encoded']
    day_of_week_encoded = encode_day_of_week(day_of_week)
    
    input_data = pd.DataFrame({
        'vehicle_count': [vehicle_count],
        'lag_1': [lag_1],
        'lag_2': [lag_2], 
        'lag_3': [lag_3],
        'hour': [hour],
        'day_of_week_encoded': [day_of_week_encoded]
    })
    
    return input_data

def encode_day_of_week(day_name):
    """Convert day name to numeric value"""
    day_mapping = {
        'Monday': 0,
        'Tuesday': 1,
        'Wednesday': 2,
        'Thursday': 3,
        'Friday': 4,
        'Saturday': 5,
        'Sunday': 6
    }
    return day_mapping.get(day_name, 1)  # Default to Tuesday if not found

def main():
    st.title("Traffic Prediction Demo")
    st.markdown("---")
    
    # Load model and data
    model = load_model()
    data = load_data()
    
    # Load current data fresh every time (no caching for real-time updates)
    current_data = load_current_data()
    
    if model is None or data is None:
        st.error("Failed to load model or data. Please check the files.")
        return
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Prediction", "Data Analysis", "Model Info"])
    
    if page == "Prediction":
        prediction_page(model, data, current_data)
    elif page == "Data Analysis":
        analysis_page(data)
    else:
        model_info_page(model, data)

def display_live_images():
    """Display CCTV images with auto-refresh capability"""
    
    # Create containers for images
    st.subheader("📹 Live CCTV Images - Sathon 12 Footbridge")
    
    # Auto-refresh toggle
    col_toggle, col_interval = st.columns([2, 1])
    with col_toggle:
        auto_refresh = st.checkbox("Auto-refresh images", value=True, help="Automatically refresh images every 30 seconds")
    with col_interval:
        refresh_interval = st.selectbox("Refresh interval", [15, 30, 60, 120], index=1, help="Seconds between refreshes")
    
    # Create placeholder containers for images
    img_container = st.container()
    status_container = st.container()
    
    # Initial load
    with img_container:
        img_col1, img_col2 = st.columns(2)
        
        with img_col1:
            st.write("**YOLO Detected Image**")
            detected_placeholder = st.empty()
        
        with img_col2:
            st.write("**Raw CCTV Image**")
            raw_placeholder = st.empty()
    
    with status_container:
        status_placeholder = st.empty()
    
    # Function to update images
    def update_images():
        # Force reload images without cache
        images = load_cctv_images()
        
        # Update detected image
        with detected_placeholder.container():
            if images['detected'] is not None:
                st.image(images['detected'], 
                        caption=f"Detected - Last updated: {images['detected_timestamp']}", 
                        use_column_width=True)
            else:
                st.error("❌ Detected image not available")
        
        # Update raw image
        with raw_placeholder.container():
            if images['raw'] is not None:
                st.image(images['raw'], 
                        caption=f"Raw - Last updated: {images['raw_timestamp']}", 
                        use_column_width=True)
            else:
                st.error("❌ Raw image not available")
        
        # Update status
        with status_placeholder.container():
            if images['detected_timestamp'] and images['raw_timestamp']:
                time_since_update = (datetime.now() - max(images['detected_timestamp'], images['raw_timestamp'])).total_seconds() / 60
                current_time = datetime.now().strftime("%H:%M:%S")
                
                if time_since_update <= 5:
                    st.success(f"🟢 Images are fresh (updated {time_since_update:.1f} minutes ago) - Last check: {current_time}")
                elif time_since_update <= 15:
                    st.warning(f"🟡 Images are moderately fresh (updated {time_since_update:.1f} minutes ago) - Last check: {current_time}")
                else:
                    st.error(f"🔴 Images may be stale (updated {time_since_update:.1f} minutes ago) - Last check: {current_time}")
            else:
                st.error("❌ Could not determine image timestamps")
        
        return images
    
    # Initial load
    images = update_images()
    
    # Auto-refresh logic
    if auto_refresh:
        # Auto-refresh using session state and rerun
        if 'last_refresh' not in st.session_state:
            st.session_state.last_refresh = time.time()
        
        current_time = time.time()
        if current_time - st.session_state.last_refresh >= refresh_interval:
            st.session_state.last_refresh = current_time
            images = update_images()
            # Small delay to prevent too frequent refreshes
            time.sleep(1)
            st.rerun()
    
    return images

def check_data_freshness():
    """Check if CSV data is fresh and return status information"""
    csv_path = 'D:/eco_project/Dashboard_update/ai-detect-traffic/bmatraffic_yolo_pipeline/src/data/sathon12_footbridge_id996_5min.csv'
    
    try:
        if not os.path.exists(csv_path):
            return {
                'status': 'error',
                'message': 'CSV file not found',
                'file_mod_time': None,
                'minutes_ago': None,
                'file_size': 0
            }
        
        # Get file information
        file_mod_time = datetime.fromtimestamp(os.path.getmtime(csv_path))
        file_size = os.path.getsize(csv_path)
        time_diff = (datetime.now() - file_mod_time).total_seconds() / 60  # minutes
        
        # Determine status based on file age
        if time_diff <= 5:
            status = 'fresh'
            status_emoji = '🟢'
        elif time_diff <= 15:
            status = 'moderate'
            status_emoji = '🟡'
        else:
            status = 'stale'
            status_emoji = '🔴'
        
        return {
            'status': status,
            'status_emoji': status_emoji,
            'message': f'File updated {time_diff:.1f} minutes ago',
            'file_mod_time': file_mod_time,
            'minutes_ago': time_diff,
            'file_size': file_size
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'message': f'Error checking file: {str(e)}',
            'file_mod_time': None,
            'minutes_ago': None,
            'file_size': 0
        }

def prediction_page(model, data, current_data):
    """Page for making predictions"""
    
    # Display live images with auto-refresh
    images = display_live_images()
    
    st.divider()
    
    # Current data selector with real-time refresh
    st.subheader("📊 Current Data for Prediction")
    st.info("💡 ข้อมูลปัจจุบันจาก CCTV ถนนสาทร")
    
    # Add data freshness check with auto-refresh
    data_status_container = st.container()
    
    # Auto-refresh data every 30 seconds
    if 'last_data_refresh' not in st.session_state:
        st.session_state.last_data_refresh = time.time()
    
    current_time_check = time.time()
    if current_time_check - st.session_state.last_data_refresh >= 30:  # 30 seconds
        st.session_state.last_data_refresh = current_time_check
        # Force reload current data
        current_data = load_current_data()
        st.rerun()
    
    # Display data freshness status
    with data_status_container:
        freshness_info = check_data_freshness()
        if freshness_info['status'] == 'fresh':
            st.success(f"{freshness_info['status_emoji']} **CSV Data Status:** Fresh - {freshness_info['message']} | Last check: {datetime.now().strftime('%H:%M:%S')}")
        elif freshness_info['status'] == 'moderate':
            st.warning(f"{freshness_info['status_emoji']} **CSV Data Status:** Moderate - {freshness_info['message']} | Last check: {datetime.now().strftime('%H:%M:%S')}")
        elif freshness_info['status'] == 'stale':
            st.error(f"{freshness_info['status_emoji']} **CSV Data Status:** Stale - {freshness_info['message']} | Last check: {datetime.now().strftime('%H:%M:%S')}")
        else:
            st.error(f"❌ **CSV Data Status:** Error - {freshness_info['message']}")
    
    col_current1, col_current2 = st.columns([1.5, 1])
    
    with col_current1:
        if current_data is not None and not current_data.empty:
            # Get the latest records from current data
            latest_records = current_data.tail(5).reset_index(drop=True)
            latest_records = latest_records.iloc[::-1]  # Reverse to show latest first
            
            selected_idx = st.selectbox(
                "เลือกข้อมูลสำหรับการทำนาย (Latest = ล่าสุด):",
                range(len(latest_records)),
                format_func=lambda x: f"{'⭐ Latest: ' if x == 0 else f'Record {x+1}: '}{latest_records.iloc[x]['timestamp']} - {latest_records.iloc[x]['vehicle_count']} vehicles"
            )
            
            # Show enhanced data status with file information
            latest_time = current_data['timestamp'].max()
            time_diff = pd.Timestamp.now() - latest_time
            minutes_ago = int(time_diff.total_seconds() / 60)
            
            # Get file metadata if available
            file_info = ""
            if hasattr(current_data, 'attrs'):
                if 'file_mod_time' in current_data.attrs:
                    file_mod_time = current_data.attrs['file_mod_time']
                    file_info = f" | File modified: {file_mod_time.strftime('%H:%M:%S')}"
                if 'records_count' in current_data.attrs:
                    file_info += f" | Total records: {current_data.attrs['records_count']}"
            

            
            # Show countdown until next refresh
            time_until_refresh = 30 - (current_time_check - st.session_state.last_data_refresh)
            if time_until_refresh > 0:
                st.caption(f" ")
        else:
            st.error("❌ Cannot load current data. Using training data samples instead.")
            # Fallback to training data samples
            latest_records = data.sample(5).reset_index(drop=True)
            selected_idx = st.selectbox(
                "เลือกข้อมูลตัวอย่างจาก Training Set:",
                range(len(latest_records)),
                format_func=lambda x: f"Sample {x+1}: {latest_records.iloc[x]['timestamp']} - {latest_records.iloc[x]['vehicle_count']} vehicles"
            )
    
    with col_current2:
        # Add label to match the dropdown structure
        st.write("")
        st.write("")

        # Make buttons match the width and positioning of dropdown
        predict_btn = st.button("Current Predict", type="primary", use_container_width=True)
        if predict_btn:
            # Force reload data before prediction to ensure latest
            fresh_data = load_current_data()
            if fresh_data is not None and not fresh_data.empty:
                fresh_records = fresh_data.tail(5).reset_index(drop=True)
                fresh_records = fresh_records.iloc[::-1]
                selected_record = fresh_records.iloc[min(selected_idx, len(fresh_records)-1)]
            else:
                selected_record = latest_records.iloc[selected_idx]
            
            st.session_state.auto_predict = True
            st.session_state.predict_record = selected_record
            st.rerun()
        
        # Add manual refresh button
    
            st.session_state.last_data_refresh = 0  # Force immediate refresh
            st.rerun()
        
        reset_btn = st.button("Reset", use_container_width=True)
        if reset_btn:
            st.session_state.auto_predict = False
            if 'predict_record' in st.session_state:
                del st.session_state.predict_record
            st.rerun()
    
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Manual Input Features")
        st.info("💡 กรอกข้อมูลเพื่อทดสอบการทำนาย")
        
        # Input form - only the features the model actually uses
        with st.form("prediction_form"):
            vehicle_count = st.number_input(
                "Current Vehicle Count",
                min_value=0,
                max_value=100,
                value=0,
                help="Number of vehicles currently observed"
            )
            
            lag_1 = st.number_input(
                "Lag 1 (Previous 5 min)",
                min_value=0,
                max_value=100,
                value=0,
                help="Vehicle count from 5 minutes ago"
            )
            
            lag_2 = st.number_input(
                "Lag 2 (Previous 10 min)",
                min_value=0,
                max_value=100,
                value=0,
                help="Vehicle count from 10 minutes ago"
            )
            
            lag_3 = st.number_input(
                "Lag 3 (Previous 15 min)",
                min_value=0,
                max_value=100,
                value=0,
                help="Vehicle count from 15 minutes ago"
            )
            
            hour = st.slider(
                "Hour of Day",
                min_value=0,
                max_value=23,
                value=12,
                help="Hour in 24-hour format"
            )
            
            day_of_week = st.selectbox(
                "Day of Week",
                options=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                index=0,
                help="Day of the week"
            )
            
            # These are not used by the model but kept for consistency
            month = 9  # Not used by model
            day = 18  # Not used by model
            minute = 0  # Not used by model
            
            submit_button = st.form_submit_button("Manual Predict", type="primary", use_container_width=True)
    
    with col2:
        st.subheader("Prediction Result")
        
        # Auto prediction from Current Data
        if hasattr(st.session_state, 'auto_predict') and st.session_state.auto_predict:
            try:
                record = st.session_state.predict_record
                
                # Extract data from the record
                vehicle_count = int(record['vehicle_count'])
                lag_1 = int(record['lag_1'])
                lag_2 = int(record['lag_2'])
                lag_3 = int(record['lag_3'])
                day_of_week = record['day_of_week']
                hour = int(record['hour'])
                
                # Prepare input data
                input_data = prepare_model_input(
                    vehicle_count, lag_1, lag_2, lag_3, day_of_week, hour, month, day, minute
                )
                
                # Make prediction
                prediction = model.predict(input_data)[0]
                
                # Display result
                st.success("🎯 Current Prediction Complete!")
                
                # Check if we have actual target value (for training data validation)
                if 'target_next_1h' in record and pd.notna(record['target_next_1h']):
                    actual_value = record['target_next_1h']
                    st.info(f"**Actual Value from Training Data:** {actual_value} vehicles")
                    
                    # Show comparison with gauge chart
                    col2_1, col2_2 = st.columns([1, 1])
                    
                    with col2_1:
                        st.metric(
                            label="Predicted Traffic (Next Hour)",
                            value=f"{prediction:.1f} vehicles",
                            delta=f"{prediction - vehicle_count:.1f} from current"
                        )
                        
                        # Calculate and show accuracy
                        error = abs(prediction - actual_value)
                        accuracy = max(0, 100 - (error / max(actual_value, 1)) * 100)
                        st.metric("Accuracy", f"{accuracy:.1f}%", f"Error: {error:.1f}")
                    
                    with col2_2:
                        # Create gauge chart showing prediction vs actual
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = prediction,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': f"Predicted: {prediction:.1f}<br>Actual: {actual_value}"},
                            gauge = {
                                'axis': {'range': [None, 50]},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [0, 10], 'color': "lightgray"},
                                    {'range': [10, 25], 'color': "gray"},
                                    {'range': [25, 50], 'color': "red"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': actual_value
                                }
                            }
                        ))
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    # For current data prediction (no target_next_1h available)
                    col2_1, col2_2 = st.columns([1, 1])
                    
                    with col2_1:
                        st.metric(
                            label="Predicted Traffic (Next Hour)",
                            value=f"{prediction:.1f} vehicles",
                            delta=f"{prediction - vehicle_count:.1f} from current"
                        )
                        
                        # Show prediction context
                        next_hour_time = record['timestamp'] + pd.Timedelta(hours=1)
                        st.info(f"**Prediction for:** {next_hour_time.strftime('%d/%m/%Y %H:%M')}")
                    
                    with col2_2:
                        # Create gauge chart
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number+delta",
                            value = prediction,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Traffic Level"},
                            delta = {'reference': vehicle_count},
                            gauge = {
                                'axis': {'range': [None, 50]},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [0, 10], 'color': "lightgray"},
                                    {'range': [10, 25], 'color': "gray"},
                                    {'range': [25, 50], 'color': "red"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 30
                                }
                            }
                        ))
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                
                # Show prediction summary with better layout
                st.divider()
                
                # Create two columns for better organization
                summary_col1, summary_col2 = st.columns([1, 1])
                
                with summary_col1:
                    st.subheader("Input Data Summary")
                    display_data = pd.DataFrame({
                        'Feature': ['Vehicle Count', 'Lag 1 (5min ago)', 'Lag 2 (10min ago)', 'Lag 3 (15min ago)', 'Hour', 'Day of Week'],
                        'Value': [str(vehicle_count), str(lag_1), str(lag_2), str(lag_3), str(hour), str(day_of_week)]
                    })
                    st.dataframe(display_data, hide_index=True, use_container_width=True)
                
                with summary_col2:
                    st.subheader("Data Source Info")
                    data_source = "Training Data" if 'target_next_1h' in record else "Current Data"
                    source_info = pd.DataFrame({
                        'Property': ['Source', 'Timestamp', 'Day of Week'],
                        'Value': [data_source, str(record['timestamp']), str(record['day_of_week'])]
                    })
                    st.dataframe(source_info, hide_index=True, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error in auto prediction: {e}")
        
        elif submit_button:
            try:
                # Prepare input data
                input_data = prepare_model_input(
                    vehicle_count, lag_1, lag_2, lag_3, day_of_week, hour, month, day, minute
                )
                
                # Make prediction
                prediction = model.predict(input_data)[0]
                
                # Display result
                st.success("🎯 Manual Prediction Complete!")
                
                # Create display for manual prediction
                col2_1, col2_2 = st.columns([1, 1])
                
                with col2_1:
                    st.metric(
                        label="Predicted Traffic (Next Hour)",
                        value=f"{prediction:.1f} vehicles",
                        delta=f"{prediction - vehicle_count:.1f} from current"
                    )
                
                with col2_2:
                    # Create gauge chart
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number+delta",
                        value = prediction,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Traffic Level"},
                        delta = {'reference': vehicle_count},
                        gauge = {
                            'axis': {'range': [None, 50]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 10], 'color': "lightgray"},
                                {'range': [10, 25], 'color': "gray"},
                                {'range': [25, 50], 'color': "red"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 30
                            }
                        }
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Show input summary with better layout
                st.divider()
                
                # Create two columns for better organization
                manual_col1, manual_col2 = st.columns([1, 1])
                
                with manual_col1:
                    st.subheader("Manual Input Summary")
                    display_data = pd.DataFrame({
                        'Feature': ['Vehicle Count', 'Lag 1 (5min ago)', 'Lag 2 (10min ago)', 'Lag 3 (15min ago)', 'Hour', 'Day of Week'],
                        'Value': [str(vehicle_count), str(lag_1), str(lag_2), str(lag_3), str(hour), str(day_of_week)]
                    })
                    st.dataframe(display_data, hide_index=True, use_container_width=True)
                
                with manual_col2:
                    st.subheader("Model Input Features")
                    st.dataframe(input_data, hide_index=True, use_container_width=True)
                    
                    # Add some info about manual input
                    st.info("💡 Manual input allows you to test different scenarios and understand model behavior.")
                
            except Exception as e:
                st.error(f"Error making prediction: {e}")
        else:
            st.info("💡 ผลลัพธ์การทำนายจากข้อมูลที่เลือก")
            
            # Show explanation about data types


def analysis_page(data):
    """Page for data analysis and visualization"""
    st.header("📊 Data Analysis")
    
    # Basic statistics
    st.subheader("Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{len(data):,}")
    with col2:
        date_range = f"{data['timestamp'].dt.date.min()} to {data['timestamp'].dt.date.max()}"
        st.metric("Date Range", date_range)
    with col3:
        st.metric("Avg Vehicle Count", f"{data['vehicle_count'].mean():.1f}")
    with col4:
        st.metric("Max Vehicle Count", data['vehicle_count'].max())
    
    # Additional statistics
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.metric("Min Vehicle Count", data['vehicle_count'].min())
    with col6:
        st.metric("Std Deviation", f"{data['vehicle_count'].std():.1f}")
    with col7:
        unique_days = data['day_of_week'].nunique()
        st.metric("Days Covered", unique_days)
    with col8:
        unique_hours = data['hour'].nunique()
        st.metric("Hours Covered", unique_hours)
    
    # Time series plot
    st.subheader("Traffic Patterns Over Time")
    
    # Allow user to select sample size
    sample_size = st.slider("Sample size for time series plot", 100, 5000, 1000, 100)
    sample_data = data.head(sample_size).copy()
    
    fig = px.line(
        sample_data, 
        x='timestamp', 
        y='vehicle_count',
        title=f"Vehicle Count Over Time (First {sample_size} records)",
        labels={'vehicle_count': 'Vehicle Count', 'timestamp': 'Time'}
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Prediction accuracy visualization (only if target_next_1h exists)
    if 'target_next_1h' in data.columns:
        st.subheader("Historical Prediction vs Actual Analysis")
        
        # Create lag-based simple prediction for comparison
        sample_for_analysis = data.head(1000).copy()
        sample_for_analysis['simple_prediction'] = sample_for_analysis['lag_1']  # Simple: assume next hour = current
        sample_for_analysis['error'] = abs(sample_for_analysis['target_next_1h'] - sample_for_analysis['simple_prediction'])
        
        col_acc1, col_acc2 = st.columns(2)
        
        with col_acc1:
            # Error distribution
            fig_error = px.histogram(
                sample_for_analysis,
                x='error',
                title="Distribution of Prediction Errors (Simple Model)",
                labels={'error': 'Absolute Error', 'count': 'Frequency'}
            )
            st.plotly_chart(fig_error, use_container_width=True)
        
        with col_acc2:
            # Actual vs Predicted scatter
            fig_scatter = px.scatter(
                sample_for_analysis.head(200),
                x='target_next_1h',
                y='simple_prediction',
                title="Actual vs Simple Prediction (First 200 records)",
                labels={'target_next_1h': 'Actual Next Hour', 'simple_prediction': 'Predicted Next Hour'}
            )
            # Add perfect prediction line
            max_val = max(sample_for_analysis['target_next_1h'].max(), sample_for_analysis['simple_prediction'].max())
            fig_scatter.add_shape(
                type="line", line=dict(dash="dash"),
                x0=0, x1=max_val, y0=0, y1=max_val
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Hourly patterns
    col1, col2 = st.columns(2)
    
    with col1:
        hourly_avg = data.groupby('hour')['vehicle_count'].mean().reset_index()
        fig = px.bar(
            hourly_avg,
            x='hour',
            y='vehicle_count',
            title="Average Traffic by Hour of Day",
            labels={'vehicle_count': 'Avg Vehicle Count', 'hour': 'Hour'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        daily_avg = data.groupby('day_of_week')['vehicle_count'].mean().reset_index()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_avg['day_of_week'] = pd.Categorical(daily_avg['day_of_week'], categories=day_order, ordered=True)
        daily_avg = daily_avg.sort_values('day_of_week')
        
        fig = px.bar(
            daily_avg,
            x='day_of_week',
            y='vehicle_count',
            title="Average Traffic by Day of Week",
            labels={'vehicle_count': 'Avg Vehicle Count', 'day_of_week': 'Day'}
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation heatmap
    st.subheader("Feature Correlations")
    numeric_cols = ['vehicle_count', 'lag_1', 'lag_2', 'lag_3', 'hour', 'day_of_week_encoded']
    if 'target_next_1h' in data.columns:
        numeric_cols.append('target_next_1h')
    corr_matrix = data[numeric_cols].corr()
    
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        title="Correlation Matrix of Numeric Features"
    )
    st.plotly_chart(fig, use_container_width=True)

def model_info_page(model, data):
    """Page showing model information"""
    st.header("Model Information")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Model Details")
        st.write("**Model Type:** XGBoost Regressor")
        st.write("**Objective:** Predict traffic count for next hour")
        st.write("**Input Features:** (6 features)")
        st.write("- vehicle_count: Current vehicle count")
        st.write("- lag_1: Vehicle count from 5 minutes ago")
        st.write("- lag_2: Vehicle count from 10 minutes ago") 
        st.write("- lag_3: Vehicle count from 15 minutes ago")
        st.write("- hour: Hour of the day (0-23)")
        st.write("- day_of_week_encoded: Day of week (0=Mon, 6=Sun)")
        st.write("")
        st.info("**Note:** This model uses traffic history, time patterns, and day-of-week information for improved accuracy.")
        
        st.subheader("Dataset Information")
        st.write(f"**Total Records:** {len(data):,}")
        st.write(f"**Features:** {len(data.columns)}")
        st.write(f"**Date Range:** {data['timestamp'].dt.date.min()} to {data['timestamp'].dt.date.max()}")
    
    with col2:
        st.subheader("Feature Statistics")
        # Show statistics for the actual model features
        model_features = ['vehicle_count', 'lag_1', 'lag_2', 'lag_3', 'hour', 'day_of_week_encoded']
        if 'target_next_1h' in data.columns:
            model_features.append('target_next_1h')
        feature_stats = data[model_features].describe()
        st.dataframe(feature_stats)
        
        st.subheader("Sample Data")
        st.dataframe(data[['timestamp'] + model_features].head(10))

if __name__ == "__main__":
    main()