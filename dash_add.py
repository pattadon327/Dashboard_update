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

# Load all models
@st.cache_resource
def load_models():
    """Load all XGBoost models for different time intervals"""
    models = {}
    model_paths = {
        '5m': 'D:/eco_project/traffic_model_5m.pkl',
        '15m': 'D:/eco_project/traffic_model_15m.pkl',
        '30m': 'D:/eco_project/traffic_model_30m.pkl',
        '1h': 'D:/eco_project/traffic_model_1h.pkl'
    }
    
    for interval, path in model_paths.items():
        try:
            with open(path, 'rb') as f:
                models[interval] = pickle.load(f)
        except Exception as e:
            st.error(f"Error loading {interval} model from {path}: {e}")
            models[interval] = None
    
    return models

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

# Load the training dataset (has all target intervals for validation)
@st.cache_data
def load_data():
    """Load the traffic training dataset with all target intervals"""
    try:
        df = pd.read_csv('D:/eco_project/traffic_dataset_with_targets.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S')
        
        # Rename lag columns to match existing code
        df = df.rename(columns={
            'lag5m': 'lag_1',
            'lag10m': 'lag_2', 
            'lag15m': 'lag_3'
        })
        
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
    # Model uses these 6 features: ['vehicle_count', 'lag5m', 'lag10m', 'lag15m', 'hour', 'day_of_week_encoded']
    day_of_week_encoded = encode_day_of_week(day_of_week)
    
    input_data = pd.DataFrame({
        'vehicle_count': [vehicle_count],
        'lag5m': [lag_1],           # lag_1 = lag5m (5 minutes ago)
        'lag10m': [lag_2],          # lag_2 = lag10m (10 minutes ago)
        'lag15m': [lag_3],          # lag_3 = lag15m (15 minutes ago)
        'hour': [hour],
        'day_of_week_encoded': [day_of_week_encoded]
    })
    
    return input_data

def make_multi_interval_predictions(models, vehicle_count, lag_1, lag_2, lag_3, day_of_week, hour, month, day, minute):
    """Make predictions for all time intervals using respective models"""
    predictions = {}
    input_data = prepare_model_input(vehicle_count, lag_1, lag_2, lag_3, day_of_week, hour, month, day, minute)
    
    for interval, model in models.items():
        if model is not None:
            try:
                prediction = model.predict(input_data)[0]
                predictions[interval] = prediction
            except Exception as e:
                st.error(f"Error making {interval} prediction: {e}")
                predictions[interval] = None
        else:
            predictions[interval] = None
    
    return predictions

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
    st.title("Multi-Interval Traffic Prediction")
    st.markdown("---")
    
    # Load models and data
    models = load_models()
    data = load_data()
    
    # Load current data fresh every time (no caching for real-time updates)
    current_data = load_current_data()
    
    if data is None:
        st.error("Failed to load data. Please check the files.")
        return
    
    # Check if at least one model loaded successfully
    if not any(model is not None for model in models.values()):
        st.error("Failed to load any models. Please check the model files.")
        return
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Prediction", "Data Analysis", "Model Info"])
    
    if page == "Prediction":
        prediction_page(models, data, current_data)
    elif page == "Data Analysis":
        analysis_page(data)
    else:
        model_info_page(models, data)

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

def prediction_page(models, data, current_data):
    """Page for making multi-interval predictions"""
    
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
        
        # Show Input Data Summary and Data Source Info below Manual Input Features
        if hasattr(st.session_state, 'auto_predict') and st.session_state.auto_predict:
            record = st.session_state.predict_record
            
            st.divider()
            
            # Create two columns for Input Data Summary and Data Source Info
            summary_col1, summary_col2 = st.columns([1, 1])
            
            with summary_col1:
                st.subheader("Input Data Summary")
                try:
                    vehicle_count_val = int(record['vehicle_count'])
                    lag_1_val = int(record['lag_1'])
                    lag_2_val = int(record['lag_2'])
                    lag_3_val = int(record['lag_3'])
                    day_of_week_val = record['day_of_week']
                    hour_val = int(record['hour'])
                    
                    display_data = pd.DataFrame({
                        'Feature': ['Vehicle Count', 'Lag 1 (5min ago)', 'Lag 2 (10min ago)', 'Lag 3 (15min ago)', 'Hour', 'Day of Week'],
                        'Value': [str(vehicle_count_val), str(lag_1_val), str(lag_2_val), str(lag_3_val), str(hour_val), str(day_of_week_val)]
                    })
                    st.dataframe(display_data, hide_index=True, use_container_width=True)
                except:
                    st.info("No current prediction data available")
            
            with summary_col2:
                st.subheader("Data Source Info")
                try:
                    data_source = "Training Data" if 'target_next_1h' in record else "Current Data"
                    source_info = pd.DataFrame({
                        'Property': ['Source', 'Timestamp', 'Day of Week'],
                        'Value': [data_source, str(record['timestamp']), str(record['day_of_week'])]
                    })
                    st.dataframe(source_info, hide_index=True, use_container_width=True)
                except:
                    st.info("No current data source info available")
    
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
                
                # Make multi-interval predictions
                predictions = make_multi_interval_predictions(
                    models, vehicle_count, lag_1, lag_2, lag_3, day_of_week, hour, month, day, minute
                )
                
                # Display result
                st.success("🎯 Current Multi-Interval Prediction Complete!")
                
                # Check if we have actual target values (for training data validation)
                target_columns = ['target_next_5m', 'target_next_15m', 'target_next_30m', 'target_next_1h']
                has_targets = any(col in record and pd.notna(record[col]) for col in target_columns)
                
                if has_targets:
                    st.info("**Comparing with Actual Values from Training Data**")
                    
                    # Display predictions for all intervals with actual values
                    interval_labels = {'5m': '5 Minutes', '15m': '15 Minutes', '30m': '30 Minutes', '1h': '1 Hour'}
                    target_mapping = {'5m': 'target_next_5m', '15m': 'target_next_15m', '30m': 'target_next_30m', '1h': 'target_next_1h'}
                    
                    # Create metrics in columns
                    metric_cols = st.columns(4)
                    
                    for i, (interval, label) in enumerate(interval_labels.items()):
                        with metric_cols[i]:
                            if predictions.get(interval) is not None:
                                pred_value = predictions[interval]
                                target_col = target_mapping[interval]
                                
                                if target_col in record and pd.notna(record[target_col]):
                                    actual_value = record[target_col]
                                    error = abs(pred_value - actual_value)
                                    accuracy = max(0, 100 - (error / max(actual_value, 1)) * 100)
                                    
                                    st.metric(
                                        label=f"Next {label}",
                                        value=f"{pred_value:.1f}",
                                        delta=f"{pred_value - vehicle_count:.1f} from current"
                                    )
                                    st.caption(f"Actual: {actual_value:.1f} | Acc: {accuracy:.1f}%")
                                else:
                                    st.metric(
                                        label=f"Next {label}",
                                        value=f"{pred_value:.1f}",
                                        delta=f"{pred_value - vehicle_count:.1f} from current"
                                    )
                                    st.caption("No actual data")
                            else:
                                st.metric(label=f"Next {label}", value="N/A")
                                st.caption("Model unavailable")
                    
                    # Show comparison chart
                    st.subheader("Predictions vs Actual Values")
                    chart_data = []
                    for interval, label in interval_labels.items():
                        if predictions.get(interval) is not None:
                            target_col = target_mapping[interval]
                            actual = record[target_col] if target_col in record and pd.notna(record[target_col]) else None
                            chart_data.append({
                                'Interval': label,
                                'Predicted': predictions[interval],
                                'Actual': actual,
                                'Current': vehicle_count
                            })
                    
                    if chart_data:
                        chart_df = pd.DataFrame(chart_data)
                        
                        fig = go.Figure()
                        fig.add_trace(go.Bar(name='Predicted', x=chart_df['Interval'], y=chart_df['Predicted'], marker_color='blue'))
                        fig.add_trace(go.Bar(name='Actual', x=chart_df['Interval'], y=chart_df['Actual'], marker_color='green'))
                        fig.add_trace(go.Scatter(name='Current', x=chart_df['Interval'], y=chart_df['Current'], 
                                               mode='lines+markers', line=dict(color='red', dash='dash')))
                        
                        fig.update_layout(title="Multi-Interval Predictions Comparison", 
                                        xaxis_title="Time Interval", yaxis_title="Vehicle Count",
                                        barmode='group', height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    
                else:
                    # For current data prediction (no target values available)
                    st.info("**Real-time Multi-Interval Predictions**")
                    
                    # Display predictions for all intervals
                    interval_labels = {'5m': '5 Minutes', '15m': '15 Minutes', '30m': '30 Minutes', '1h': '1 Hour'}
                    
                    # Create metrics in columns
                    metric_cols = st.columns(4)
                    
                    for i, (interval, label) in enumerate(interval_labels.items()):
                        with metric_cols[i]:
                            if predictions.get(interval) is not None:
                                pred_value = predictions[interval]
                                st.metric(
                                    label=f"Next {label}",
                                    value=f"{pred_value:.1f}",
                                    delta=f"{pred_value - vehicle_count:.1f} from current"
                                )
                                
                                # Show prediction time
                                if interval == '5m':
                                    pred_time = record['timestamp'] + pd.Timedelta(minutes=5)
                                elif interval == '15m':
                                    pred_time = record['timestamp'] + pd.Timedelta(minutes=15)
                                elif interval == '30m':
                                    pred_time = record['timestamp'] + pd.Timedelta(minutes=30)
                                else:  # 1h
                                    pred_time = record['timestamp'] + pd.Timedelta(hours=1)
                                
                                st.caption(f"Time: {pred_time.strftime('%H:%M')}")
                            else:
                                st.metric(label=f"Next {label}", value="N/A")
                                st.caption("Model unavailable")
                    
                    # Show prediction chart
                    st.subheader("Multi-Interval Predictions")
                    chart_data = []
                    for interval, label in interval_labels.items():
                        if predictions.get(interval) is not None:
                            chart_data.append({
                                'Interval': label,
                                'Predicted': predictions[interval]
                            })
                    
                    if chart_data:
                        chart_df = pd.DataFrame(chart_data)
                        
                        fig = go.Figure()
                        fig.add_trace(go.Bar(name='Predicted', x=chart_df['Interval'], y=chart_df['Predicted'], marker_color='blue'))
                        
                        fig.update_layout(title="Multi-Interval Predictions", 
                                        xaxis_title="Time Interval", yaxis_title="Vehicle Count",
                                        height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Show main gauge for 1 hour prediction (most important)
                    if predictions.get('1h') is not None:
                        col_gauge1, col_gauge2 = st.columns(2)
                        with col_gauge1:
                            # Create gauge chart for 1h prediction
                            fig = go.Figure(go.Indicator(
                                mode = "gauge+number+delta",
                                value = predictions['1h'],
                                domain = {'x': [0, 1], 'y': [0, 1]},
                                title = {'text': "1-Hour Prediction"},
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
                        
                        with col_gauge2:
                            # Create gauge chart for 5m prediction (shortest term)
                            if predictions.get('5m') is not None:
                                fig = go.Figure(go.Indicator(
                                    mode = "gauge+number+delta",
                                    value = predictions['5m'],
                                    domain = {'x': [0, 1], 'y': [0, 1]},
                                    title = {'text': "5-Minute Prediction"},
                                    delta = {'reference': vehicle_count},
                                    gauge = {
                                        'axis': {'range': [None, 50]},
                                        'bar': {'color': "darkgreen"},
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
                
            except Exception as e:
                st.error(f"Error in auto prediction: {e}")
        
        elif submit_button:
            try:
                # Make multi-interval predictions
                predictions = make_multi_interval_predictions(
                    models, vehicle_count, lag_1, lag_2, lag_3, day_of_week, hour, month, day, minute
                )
                
                # Display result
                st.success("🎯 Manual Multi-Interval Prediction Complete!")
                
                # Display predictions for all intervals
                interval_labels = {'5m': '5 Minutes', '15m': '15 Minutes', '30m': '30 Minutes', '1h': '1 Hour'}
                
                # Create metrics in columns
                metric_cols = st.columns(4)
                
                for i, (interval, label) in enumerate(interval_labels.items()):
                    with metric_cols[i]:
                        if predictions.get(interval) is not None:
                            pred_value = predictions[interval]
                            st.metric(
                                label=f"Next {label}",
                                value=f"{pred_value:.1f}",
                                delta=f"{pred_value - vehicle_count:.1f} from current"
                            )
                        else:
                            st.metric(label=f"Next {label}", value="N/A")
                            st.caption("Model unavailable")
                
                # Show prediction chart
                st.subheader("Manual Multi-Interval Predictions")
                chart_data = []
                for interval, label in interval_labels.items():
                    if predictions.get(interval) is not None:
                        chart_data.append({
                            'Interval': label,
                            'Predicted': predictions[interval]
                        })
                
                if chart_data:
                    chart_df = pd.DataFrame(chart_data)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(name='Predicted', x=chart_df['Interval'], y=chart_df['Predicted'], marker_color='blue'))
                    
                    fig.update_layout(title="Manual Multi-Interval Predictions", 
                                    xaxis_title="Time Interval", yaxis_title="Vehicle Count",
                                    height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Create display for detailed analysis
                col2_1, col2_2 = st.columns([1, 1])
                
                with col2_1:
                    # Show 1-hour prediction as main metric (if available)
                    if predictions.get('1h') is not None:
                        st.metric(
                            label="Main: Next Hour",
                            value=f"{predictions['1h']:.1f} vehicles",
                            delta=f"{predictions['1h'] - vehicle_count:.1f} from current"
                        )
                    else:
                        st.metric(label="Main: Next Hour", value="N/A")
                        st.caption("1h model unavailable")
                    
                    # Show additional prediction summary
                    available_predictions = [interval for interval, pred in predictions.items() if pred is not None]
                    st.info(f"Available models: {', '.join(available_predictions)}")
                
                with col2_2:
                    # Create gauge chart for 1h prediction (if available)
                    if predictions.get('1h') is not None:
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number+delta",
                            value = predictions['1h'],
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "1-Hour Prediction"},
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
                    else:
                        st.info("1-hour model not available for gauge chart")
                
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
    
    # Multi-interval target analysis
    target_columns = ['target_next_5m', 'target_next_15m', 'target_next_30m', 'target_next_1h']
    available_targets = [col for col in target_columns if col in data.columns]
    
    if available_targets:
        st.subheader("Multi-Interval Target Analysis")
        
        # Show target statistics
        st.write("**Target Variables Statistics:**")
        target_stats = data[available_targets].describe()
        st.dataframe(target_stats)
        
        # Create comparison visualizations
        sample_for_analysis = data.head(1000).copy()
        
        # Simple baseline predictions for comparison
        baseline_predictions = {
            'target_next_5m': 'vehicle_count',    # Current = next 5min (very simple)
            'target_next_15m': 'lag_1',          # 5min ago = next 15min
            'target_next_30m': 'lag_1',          # 5min ago = next 30min  
            'target_next_1h': 'lag_1'            # 5min ago = next 1h
        }
        
        # Calculate errors for available targets
        col_error1, col_error2 = st.columns(2)
        
        with col_error1:
            # Error comparison chart
            error_data = []
            for target_col in available_targets:
                if target_col in baseline_predictions:
                    baseline_col = baseline_predictions[target_col]
                    if baseline_col in sample_for_analysis.columns:
                        errors = abs(sample_for_analysis[target_col] - sample_for_analysis[baseline_col])
                        error_data.extend([{
                            'Interval': target_col.replace('target_next_', '').replace('m', ' min').replace('h', ' hour'),
                            'Error': error,
                            'Target_Value': sample_for_analysis[target_col].iloc[i],
                            'Baseline_Pred': sample_for_analysis[baseline_col].iloc[i]
                        } for i, error in enumerate(errors) if pd.notna(error)])
            
            if error_data:
                error_df = pd.DataFrame(error_data)
                fig_error = px.box(
                    error_df,
                    x='Interval',
                    y='Error', 
                    title="Baseline Prediction Errors by Interval"
                )
                st.plotly_chart(fig_error, use_container_width=True)
        
        with col_error2:
            # Target correlation matrix
            corr_targets = data[available_targets + ['vehicle_count']].corr()
            fig_corr = px.imshow(
                corr_targets,
                text_auto=True,
                title="Target Variables Correlation"
            )
            st.plotly_chart(fig_corr, use_container_width=True)
        
        # Multi-interval comparison over time
        st.subheader("Multi-Interval Targets Over Time")
        
        # Select a smaller sample for visualization
        viz_sample = sample_for_analysis.head(200).copy()
        
        # Melt the data for easier plotting
        target_data = viz_sample[['timestamp'] + available_targets].melt(
            id_vars=['timestamp'], 
            var_name='Target_Type', 
            value_name='Vehicle_Count'
        )
        target_data['Target_Type'] = target_data['Target_Type'].str.replace('target_next_', '').str.replace('m', ' min').str.replace('h', ' hour')
        
        fig_multi = px.line(
            target_data,
            x='timestamp',
            y='Vehicle_Count',
            color='Target_Type',
            title="All Target Intervals Over Time (First 200 records)"
        )
        fig_multi.update_layout(height=400)
        st.plotly_chart(fig_multi, use_container_width=True)

    
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
    
    # Enhanced Correlation heatmap
    st.subheader("Feature Correlations")
    numeric_cols = ['vehicle_count', 'lag_1', 'lag_2', 'lag_3', 'hour', 'day_of_week_encoded']
    
    # Add all available target columns
    target_columns = ['target_next_5m', 'target_next_15m', 'target_next_30m', 'target_next_1h']
    available_targets = [col for col in target_columns if col in data.columns]
    numeric_cols.extend(available_targets)
    
    corr_matrix = data[numeric_cols].corr()
    
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto", 
        title="Correlation Matrix: Features and All Target Intervals"
    )
    fig.update_layout(height=600)  # Make it taller to accommodate more variables
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