import base64
import flash
import django
import django1
import django2
from multiprocessing import Process, Queue
from services.zone import ( insert_zone, get_zone, edit_zone, delete_zone, add_alert, get_zone_alerts, edit_alert, delete_alerts, get_alerts_notifications,)

from multiprocessing import Process, Queue
from services.zone import ( insert_zone, get_zone, edit_zone, delete_zone, add_alert, get_zone_alerts, edit_alert, delete_alerts, get_alerts_notifications,)

from services.flame_alerts import get_open_state_alerts, change_alert_state
import docker
import yaml
import pymongo
from bson import ObjectId
from cam_status import cameras_status
from fastapi import (
    Depends,
    FastAPI,
    BackgroundTasks,
    File,
    Form,
    Header,
    HTTPException,
    UploadFile,
)
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.docs import (
    get_redoc_html,
    get_swagger_ui_html,
    get_swagger_ui_oauth2_redirect_html,
)
from models import (
    AlprEntities,
    BucketUpload,
    Camera,
    ServiceContainer,
    ServicesConfigs,
    Homography,
    Feedback,
    NetworkScan,
    BulkSensors,
    AuthenticateSensor,
)

from core.config import sensors, proj_settings
from pagination import Pagination
from services.spaces_apis import router as spaces_apis
from services.mail_room import router as mailroom_apis
from services.mtel_ids import get_mtel_ids_from_db
from services.high_value_asset import insert_ids, get_ids
from services.avgm_integration import get_all_avgm_records
from services.project_settings import router as project_settings_apis
from services.ai_models import router as ai_models_apis
from services.integration_profile import router as integration_profile_apis
from starlette.middleware.cors import CORSMiddleware
from core.categorise_data import categorised_config_data
from starlette.requests import Request
from utils import (
    auth_user,
    get_camera_preview,
    populate_alprEntities,
    sanitize,
    authentication,
    validate_ip,
    check_resource_permissions,
)
from core.logger import logging
import cv2
import numpy as np

app = FastAPI(title="Config Service", docs_url=None, redoc_url=None)
try:
    app.mount("/static", StaticFiles(directory="/static"), name="static")
except Exception:
    pass

origins = ["*", "http://localhost:8080"]

MAX_CAM_NAME_LEN = 30
MAX_CAM_USERNAME_LEN = 30
MAX_CAM_PASSWORD_LEN = 50
MAX_CAM_URL_LEN = 120

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_event_handler("startup", create_start_app_handler(app=app))

app.include_router(spaces_apis, tags=["spaces"], prefix="/spaces")
app.include_router(mailroom_apis, tags=["mailroom"], prefix="/mailroom")
app.include_router(
    zone_statistics_apis, tags=["zones", "stats"], prefix="/zone-statistics"
)
app.include_router(project_settings_apis, tags=["projects"], prefix="/projects")
app.include_router(ai_models_apis, tags=["AI Models"], prefix="/ai_models")
app.include_router(
    integration_profile_apis, tags=["integration_profile"], prefix="/integration"
)
db_host = os.getenv("MONGODB_HOSTNAME", "mongo")
db_port = "27017"
database = os.environ["database"]
client = pymongo.MongoClient(f"mongodb://{db_host}:{db_port}/")
db = client[database]
planCol = db["FloorPlans"]
obscure_mask_col = db["ObscuringMasks"]
services_settings = db["ServicesSettings"]
ai_classes_col = db["ModelClasses"]
alpr_entities_col = db["AlprEntities"]
annotation = db["Annotation"]
feedback = db["Feedback"]
other_classes_col = db["CustomModels"]
custom_keys_col = db["CustomKeys"]
ai_classes_status = False
ai_classes = None
ENABLE_IMAGE_PREVIEW_TIMEOUT = get_env_variable("enable_image_preview_timeout", True)
IMAGE_PREVIEW_TIMEOUT = get_env_variable("image_preview_timeout", 3)
non_empty = any(Path(f"/data/{database}").iterdir())
if not proj_settings.find_one() and non_empty:
    command = f"mongorestore --host {db_host} --port {db_port} --db {database}"
    command += f"  --drop /data/{database}"
    os.system(command)
# Get rid of this once integrate in dashboard
if not services_settings.find_one():
    services_settings.insert_one(all_configs)
else:
    last_config = services_settings.find_one({}, sort=[("_id", pymongo.DESCENDING)])
    services_settings.replace_one(last_config, all_configs)


def get_all_models():
    models = os.listdir("resources/")
    return models


def get_object_details():
    object_details = []
    file_path = str(list(Path("resources/").glob("*.txt"))[0])
    with open(file_path, "r") as f:
        names = f.read()
    classes_slugs = [x.strip() for x in list(filter(None, names.split("\n")))]

    fill_level_file_path = str(list(Path("resources/").glob("second_stage*.txt"))[0])
    with open(fill_level_file_path, "r") as f:
        fill_levels = f.read()

    sack_color_file_path = str(list(Path("resources/").glob("third_stage*.txt"))[0])
    with open(sack_color_file_path, "r") as f:
        sack_colors = f.read()
    sack_colors = [x.strip() for x in sack_colors.split(";")]
    fill_levels = [x.strip() for x in fill_levels.split(";")]
    non_container_classes = ["person"]
    non_container_classes.extend(all_configs["enricher"]["vehicle_classes"])
    classes_slugs = list(set(classes_slugs) - set(non_container_classes))
    for obj in classes_slugs:
        object_details.append(
            {"class": obj, "fill_levels": fill_levels, "colors": sack_colors}
        )
    return object_details


def is_new_classes_structure(classes):
    if not classes:
        return False
    if classes.get("class_slug", None) is not None:
        ai_classes_col.delete_many({})
        return False
    else:
        return True


def load_detector_ai_classes():
    try:
        two_ = all_configs["superdetector"]["second_stage"]
        one_ = ai_classes_col.find_one(
            {}, {"_id": 0}, sort=[("_id", pymongo.DESCENDING)]
        )
        if is_new_classes_structure(classes=one_):
            return True
        model_classes = {}
        models = get_all_models()
        fill_classes = []
        sack_color_classes = []
        for model in models:
            try:
                file_path = str(
                    list(Path(f"resources/{model}/detector").glob("*.txt"))[0]
                )
                with open(file_path, "r") as f:
                    names = f.read()
                classes_slugs = [
                    x.strip() for x in list(filter(None, names.split("\n")))
                ]
                classes_display_name = [slug.split("_")[0] for slug in classes_slugs]
                if two_ and os.path.isdir(f"resources/{model}/classifier/fill_level"):
                    fill_file_path = str(
                        list(
                            Path(f"resources/{model}/classifier/fill_level").glob(
                                "*.txt"
                            )
                        )[0]
                    )
                    with open(fill_file_path, "r") as f2:
                        fill_names = f2.read()
                    fill_classes = [
                        x.strip() for x in list(filter(None, fill_names.split(";")))
                    ]
                if os.path.isdir(f"resources/{model}/classifier/color"):
                    sack_color_file_path = str(
                        list(Path(f"resources/{model}/classifier/color").glob("*.txt"))[
                            0
                        ]
                    )
                    with open(sack_color_file_path, "r") as f2:
                        sack_color_names = f2.read()
                    sack_color_classes = [
                        x.strip()
                        for x in list(filter(None, sack_color_names.split(";")))
                    ]
                classes = {
                    model: {
                        "class_slug": classes_slugs,
                        "class_display_name": classes_display_name,
                        "fill_classes": fill_classes,
                        "sack_colors": sack_color_classes,
                    }
                }
                model_classes.update(classes)
            except Exception:
                logging.exception(f"Detector file for model {model} not found on s3")
                continue
        model_classes.update({"created_at": time.time(), "updated_at": time.time()})
        ai_classes_col.insert_one(model_classes)
        logging.info("Updated Ai classes for detector.")
    except Exception:
        logging.debug("Unable to fetch ai classes")


load_detector_ai_classes()


def get_value_from_data(data):
    if type(data) is list:
        if data[1] == "int":
            return int(data[0])
        elif data[1] == "str" or data[1] == "ip":
            return str(data[0])
        elif data[1] == "bool":
            if data[0] is True:
                return True
            return False
        elif data[1] == "float":
            return float(data[0])
        return data[0]
    elif type(data) is dict:
        return get_values_from_nested_data(data=data)


def get_values_from_nested_data(data):
    for item, value in data.items():
        if type(value) is dict:
            get_values_from_nested_data(data=value)
            continue
        if data[str(item)][1] == "int":
            data[str(item)] = int(data[str(item)][0])
        elif data[str(item)][1] == "str" or data[str(item)][1] == "ip":
            data[str(item)] = str(data[str(item)][0])
        elif data[str(item)][1] == "bool":
            if data[str(item)][0] is True:
                data[str(item)] = True
            data[str(item)] = False
        elif data[str(item)][1] == "float":
            data[str(item)] = float(data[str(item)][0])
        else:
            pass
    return data


def validate_time(schedule):
    rules = [
        str(schedule["start_time"])
        == str(
            datetime.datetime.strptime((schedule["start_time"]), "%Y-%m-%d %H:%M:%S")
        ),
        str(schedule["end_time"])
        == str(datetime.datetime.strptime(schedule["end_time"], "%Y-%m-%d %H:%M:%S")),
    ]
    if all(rules):
        return True
    else:
        return False


def validate_schedule(schedules):
    for schedule in schedules.values():
        for sub_schedules in schedule:
            if sub_schedules != []:
                if sub_schedules["start_time"] < sub_schedules[
                    "end_time"
                ] and validate_time(schedule=sub_schedules):
                    continue
                else:
                    return False
    return True


def validate_cam_params(username: str, password: str, url: str, name: str = None):
    """
    Returns true,none if given cam params are valid else false,msg.
    """
    if username and len(username) > MAX_CAM_USERNAME_LEN:
        return (False, f"Max allowed username length : {MAX_CAM_USERNAME_LEN}")
    elif password and len(password) > MAX_CAM_PASSWORD_LEN:
        return (False, f"Max allowed password length : {MAX_CAM_PASSWORD_LEN}")
    elif len(url) > MAX_CAM_URL_LEN:
        return (False, f"Max allowed url length ; {MAX_CAM_URL_LEN}")

    if name:
        if len(name) > MAX_CAM_NAME_LEN:
            return (False, f"Max allowed name length : {MAX_CAM_NAME_LEN}")

    if "/videos" not in url:
        try:
            validators = urlparse(url)
            if all([validators.path]):
                return (True, None)
            else:
                return (False, "Invalid Url")
        except Exception:
            return (False, "Invalid Url")

    return True, None


@app.get("/settings", name="Get Settings")
async def get_settings(
        request: Request,
        is_authenticated: bool = Depends(authentication),
):
    """
    Get Project Settings.

    * **URL**
        `localhost:8000/settings`


    * **Method**
        `[GET]`

    * **Headers**
        * `'accept: application/json'`
        * `Authorization': 'bearer {token}'`

    * **Response**
        ::

            HTTP/1.1 200 OK
            Date: Thu, 24 Feb 2011 12:36:30 GMT
            Status: 200 OK
            Content-Type: application/json
            Examples
            {"message": "success", "data": {Settings}}

    * **Error Response**
        ::

            HTTP/1.1 401 UnAuthorized
            Status: 401 UnAuthorized
            Content-Type: application/json
            Reasons:
            Provided access token is not valid.

    * **Error Response**
    * **Error Response**
        ::

            HTTP/1.1 404 Settings not found
            Status: 404
            Content-Type: application/json
            Reasons:
            Settings does not exist in database.
    """
    if not is_authenticated:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not check_resource_permissions(request=request):
        raise HTTPException(
            status_code=403,
            detail="You don't have enough permissions to access this resource.",
        )
    settings = list(proj_settings.find({}, {"_id": 0}))
    if settings:
        return {"message": "success", "data": settings}
    else:
        raise HTTPException(status_code=404, detail="Settings not found")


@app.get("/floor-plan", name="Get Floor Plan")
def get_floor_plan(
        floor_plan_id: str,
        request: Request,
        is_authenticated: bool = Depends(authentication),
):
    """
    Get Floor Plan

    * **URL**
        `localhost:8000/floor-plan`


    * **Method**
        `[GET]`

    * **Headers**
        * `'accept: application/json'`
        * `Authorization': 'bearer {token}'`

    * **Request Body**
        * `floor_plan_id (Required)`

            `string`

    * **Response**
        ::

            HTTP/1.1 200 OK
            Date: Thu, 24 Feb 2011 12:36:30 GMT
            Status: 200 OK
            Content-Type: application/json
            Examples
            {"message": "success", "floor-plan": floor-plan-image}

    * **Error Response**
        ::

            HTTP/1.1 401 UnAuthorized
            Status: 401 UnAuthorized
            Content-Type: application/json
            Reasons:
            Provided access token is not valid.

    * **Error Response**
        ::

            HTTP/1.1 404 Settings not found
            Status: 404
            Content-Type: application/json
            Reasons:
            Settings does not exist in database.

    * **Error Response**
        ::

            HTTP/1.1 422 Validation Error
            Status: 422 Validation Error
            Content-Type: application/json
            Example:
            { "detail": [ { "loc": [ "string" ], "msg": "string", "type": "string" } ] }
    """
    if not is_authenticated:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not check_resource_permissions(request=request):
        raise HTTPException(
            status_code=403,
            detail="You don't have enough permissions to access this resource.",
        )
    floor_plan_id = sanitize(floor_plan_id)
    try:
        floorplan = planCol.find_one(
            {"_id": ObjectId(floor_plan_id)},
            {"_id": 0},
            sort=[("_id", pymongo.DESCENDING)],
        )
        return {"message": "success", "floor-plan": floorplan}
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Floor plan not found {e}")


@app.post("/floor-plan", name="Save Floor Plan")
def save_floor_plan(
        request: Request,
        floor_plan: UploadFile = File(...),
        orientation: str = Form(default=None),
        floorplan_scale: int = Form(default=1),
        is_authenticated: bool = Depends(authentication),
):
    """
    Save Floor Plan

    * **URL**
        `localhost:8000/floor-plan`


    * **Method**
        `[POST]`

    * **Headers**
        * `'accept: application/json'`
        * `Authorization': 'bearer {token}'`

    * **Request Body**
        * `floor_plan (Required)`

            `Image file`

    * **Response**
        ::

            HTTP/1.1 200 OK
            Date: Thu, 24 Feb 2011 12:36:30 GMT
            Status: 200 OK
            Content-Type: application/json
            Examples
            {"message": "success", "floor-plan": floor-plan-id}

    * **Error Response**
        ::

            HTTP/1.1 401 UnAuthorized
            Status: 401 UnAuthorized
            Content-Type: application/json
            Reasons:
            Provided access token is not valid.

    * **Error Response**
        ::

            HTTP/1.1 404 Settings not found
            Status: 404
            Content-Type: application/json
            Reasons:
            Settings does not exist in database.

    * **Error Response**
        ::

            HTTP/1.1 422 Validation Error
            Status: 422 Validation Error
            Content-Type: application/json
            Example:
            { "detail": [ { "loc": [ "string" ], "msg": "string", "type": "string" } ] }
    """
    if not is_authenticated:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not check_resource_permissions(request=request):
        raise HTTPException(
            status_code=403,
            detail="You don't have enough permissions to access this resource.",
        )
    byto = floor_plan.file.read()
    sitePlan = base64.b64encode(byto)
    img = Image.open(io.BytesIO(byto))
    width, height = img.size
    floor_plan_id = planCol.insert_one(
        {
            "created_at": datetime.datetime.now(),
            "sitePlan": sitePlan,
            "orientation": orientation,
            "floorplan_scale": floorplan_scale,
            "floorplan_width": width,
            "floorplan_height": height,
        }
    ).inserted_id
    return {"message": "success", "floor_plan_id": str(floor_plan_id)}


@app.post("/add-settings", name="Set Settings")
def add_settings(
        request: Request,
        siteName: str = Form(...),
        version: str = Form(...),
        sitePlan: str = Form(...),
        horizontal_offset: float = Form(...),
        vertical_offset: float = Form(...),
        is_authenticated: bool = Depends(authentication),
):
    """
    Set Project Settings.

    * **URL**
        `localhost:8000/add-settings`


    * **Method**
        `[POST]`

    * **Headers**
        * `'accept: application/json'`
        * `Authorization': 'bearer {token}'`

    * **Request Body**
        * `siteName (Required)`

            `string`

        * `version (Required)`

            `string`

        * `sitePlan (Required)`

            `string`

        * `horizontal_offset (Required)`

            `float`

        * `vertical_offset (Required)`

            `float`

    * **Response**
        ::

            HTTP/1.1 200 OK
            Date: Thu, 24 Feb 2011 12:36:30 GMT
            Status: 200 OK
            Content-Type: application/json
            Examples:
            {"message": "Config Added successfully", "data": {db_id}}

    * **Error Response**
        ::

            HTTP/1.1 401 UnAuthorized
            Status: 401 UnAuthorized
            Content-Type: application/json
            Reasons:
            Provided access token is not valid.

    * **Error Response**
        ::

            HTTP/1.1 422 Validation Error
            Status: 422 Validation Error
            Content-Type: application/json
            Example:
            { "detail": [ { "loc": [ "string" ], "msg": "string", "type": "string" } ] }
    """
    if not is_authenticated:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not check_resource_permissions(request=request):
        raise HTTPException(
            status_code=403,
            detail="You don't have enough permissions to access this resource.",
        )
    siteName = sanitize(siteName)
    version = sanitize(version)
    sitePlan = sanitize(sitePlan)
    config = {
        "siteName": siteName,
        "sitePlan": sitePlan,
        "version": version,
        "horizontal_offset": horizontal_offset,
        "vertical_offset": vertical_offset,
    }
    setting = proj_settings.find_one({"siteName": siteName})
    if setting:
        raise HTTPException(
            status_code=403,
            detail="Resource already exists",
        )
    settings_id = proj_settings.insert_one(config).inserted_id
    return {"message": "Config Added successfully", "data": str(settings_id)}


@app.post("/homography", name="Set Homography")
def homography_matrix(
        request: Request, pts: Homography, is_authenticated: bool = Depends(authentication)
):
    """
    Set Homography.

    * **URL**
        `localhost:8000/homography`

    * **Method**
        `[POST]`

    * **Headers**
        * `'accept: application/json'`
        * `Authorization': 'bearer {token}'`

    * **Request Body**
        * `homography (Required)`

            `dict`

        ::

            Example
            {
                "src_points": ["string"],
                "dst_points": ["string"]
            }

    * **Response**
        ::

            HTTP/1.1 200 OK
            Date: Thu, 24 Feb 2011 12:36:30 GMT
            Status: 200 OK
            Content-Type: application/json
            Examples
            return {"message": "success", "data": {list of points}}

    * **Error Response**
        ::

            HTTP/1.1 400 Src or Dst points Invalid
            Status: 400
            Content-Type: application/json
            Reasons:
            When points are less than 4.

    * **Error Response**
        ::

            HTTP/1.1 401 UnAuthorized
            Status: 401 UnAuthorized
            Content-Type: application/json
            Reasons:
            Provided access token is not valid.

    * **Error Response**
        ::

            HTTP/1.1 422 Validation Error
            Status: 422 Validation Error
            Content-Type: application/json
            Example:
            { "detail": [ { "loc": [ "string" ], "msg": "string", "type": "string" } ] }
    """
    if not is_authenticated:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not check_resource_permissions(request=request):
        raise HTTPException(
            status_code=403,
            detail="You don't have enough permissions to access this resource.",
        )
    pts = pts.dict()
    if len(pts["src_points"]) >= 4 and len(pts["dst_points"]) >= 4:
        src_pts = np.array(pts["src_points"]).reshape(-1, 1, 2)
        dst_pts = np.array(pts["dst_points"]).reshape(-1, 1, 2)
        matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return {"message": "success", "data": matrix.tolist()}

    else:
        raise HTTPException(status_code=400, detail="Src or Dst points Invalid")


@app.get("/cam-status", name="Camera Statuses")
async def get_cam_status(
        request: Request, is_authenticated: bool = Depends(authentication)
):
    """
    Get all camera connectivity statuses.
    ie ping info and opencv connectivity status.

    * **URL**
        `localhost:8000/cam-status`


    * **Method**
        `[GET]`

    * **Headers**
        * `'accept: application/json'`
        * `Authorization': 'bearer {token}'`

    * **Response**
        ::

            HTTP/1.1 200 OK
            Date: Thu, 24 Feb 2011 12:36:30 GMT
            Status: 200 OK
            Content-Type: application/json
            Examples
            return {"cam_status": {Status data of all cams}}

    * **Error Response**
        ::

            HTTP/1.1 401 UnAuthorized
            Status: 401 UnAuthorized
            Content-Type: application/json
            Reasons:
            Provided access token is not valid.
    """
    if not is_authenticated:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not check_resource_permissions(request=request):
        raise HTTPException(
            status_code=403,
            detail="You don't have enough permissions to access this resource.",
        )
    return {"cam_status": cameras_status(sensors)}


@app.get("/ai/classes", name="Get Ai Classes")
async def get_sensors_classes(
        request: Request, is_authenticated: bool = Depends(authentication)
):
    """
    Get all Ai Classes

    * **URL**
        `localhost:8000/ai/classes`


    * **Method**
        `[GET]`

    * **Headers**
        * `'accept: application/json'`
        * `Authorization': 'bearer {token}'`

    * **Response**
        ::

            HTTP/1.1 200 OK
            Date: Thu, 24 Feb 2011 12:36:30 GMT
            Status: 200 OK
            Content-Type: application/json
            Examples
            return {"message": "success", "data": classes}

    * **Error Response**
        ::

            HTTP/1.1 401 UnAuthorized
            Status: 401 UnAuthorized
            Content-Type: application/json
            Reasons:
            Provided access token is not valid.
    """
    if not is_authenticated:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not check_resource_permissions(request=request):
        raise HTTPException(
            status_code=403,
            detail="You don't have enough permissions to access this resource.",
        )
    classes = ai_classes_col.find_one(
        {}, {"_id": 0}, sort=[("_id", pymongo.DESCENDING)]
    )

    return {"message": "success", "data": classes}


@app.get("/ai/names", name="Get Ai Names")
async def get_ai_names(
        request: Request, is_authenticated: bool = Depends(authentication)
):
    """
    This api is responsible to get information
    of class names of 2nd stage models.

    * **URL**
        `localhost:8000/ai/names`

    * **Method**
        `[GET]`

    * **Headers**
        * `'accept: application/json'`
        * `Authorization': 'bearer {token}'`

    * **Response**
        ::

            HTTP/1.1 200 OK
            Date: Thu, 24 Feb 2011 12:36:30 GMT
            Status: 200 OK
            Content-Type: application/json
            Examples:
            return return {"message": "success", "data": second_stage_names}

    * **Error Response**
        ::

            HTTP/1.1 401 UnAuthorized
            Status: 401 UnAuthorized
            Content-Type: application/json
            Reasons:
            Provided access token is not valid.

    * **Error Response**
        ::

            HTTP/1.1 404 File not found
            Status: 401 Not Found
            Content-Type: application/json
            Reasons:
            Second model names file does not exist.
    """
    if not is_authenticated:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not check_resource_permissions(request=request):
        raise HTTPException(
            status_code=403,
            detail="You don't have enough permissions to access this resource.",
        )
    if Path(second_stage_names_path).exists():
        with open(second_stage_names_path, "r") as f:
            names = f.read()
        second_stage_names = [x.strip() for x in list(filter(None, names.split("\n")))]
    else:
        raise HTTPException(status_code=404, detail="File not found")
    return {"message": "success", "data": second_stage_names}


@app.put("/ai/classes", name="Set Ai Classes")
async def update_config_ai(
        request: Request,
        classes: dict,
        is_authenticated: bool = Depends(authentication),
):
    """
    Set Project AiClasses.

    * **URL**
        `localhost:8000/ai/classes`

    * **Method**
        `[POST]`

    * **Headers**
        * `'accept: application/json'`
        * `Authorization': 'bearer {token}'`

    * **Request Body**
        * `classes (Required)`

            `dict`

        ::

            Example
            {
                "class_slug": "string",
                "class_display_name": "string"
            }

    * **Response**
        ::

            HTTP/1.1 200 OK
            Date: Thu, 24 Feb 2011 12:36:30 GMT
            Status: 200 OK
            Content-Type: application/json
            Examples:
            {"message": "AI classes updated successfully", "data": {db_id}}

    * **Error Response**
        ::

            HTTP/1.1 401 UnAuthorized
            Status: 401 UnAuthorized
            Content-Type: application/json
            Reasons:
            Provided access token is not valid.

    * **Error Response**
        ::

            HTTP/1.1 422 Validation Error
            Status: 422 Validation Error
            Content-Type: application/json
            Example:
            { "detail": [ { "loc": [ "string" ], "msg": "string", "type": "string" } ] }
    """
    if not is_authenticated:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not check_resource_permissions(request=request):
        raise HTTPException(
            status_code=403,
            detail="You don't have enough permissions to access this resource.",
        )
    classes_document = ai_classes_col.find_one({}, sort=[("_id", pymongo.DESCENDING)])
    if classes_document:
        ai_classes_col.delete_many(filter={"_id": {"$ne": classes_document["_id"]}})
        for key, value in classes.items():
            sanitize(value.get("class_slug", ""))
            sanitize(value.get("class_display_name", ""))
            sanitize(value.get("fill_classes", ""))
            sanitize(value.get("sack_color", ""))
            updated_classes = {"$set": {key: value}}
        ai_classes_col.update_one(
            filter={"_id": classes_document["_id"]}, update=updated_classes
        )
        return {
            "code": 200,
            "message": "AI classes updated successfully",
            "data": {"id": str(classes_document["_id"])},
        }
    else:
        raise HTTPException(
            status_code=404,
            detail="classes not found",
        )


@app.get("/alpr/entities", name="Get Alpr Entities")
async def get_alpr_entities(
        request: Request, is_authenticated: bool = Depends(authentication)
):
    """
    This api is responsible to get known license plates.
    If there are no entry for known license plate in the db, this will
    add default known stickers to the db.
    ALPR service requires known entries to compare the detection.

    * **URL**
        `localhost:8000/alpr/entities`


    * **Method**
        `[GET]`

    * **Headers**
        * `'accept: application/json'`
        * `Authorization': 'bearer {token}'`

    * **Response**
        ::

            HTTP/1.1 200 OK
            Date: Thu, 24 Feb 2011 12:36:30 GMT
            Status: 200 OK
            Content-Type: application/json
            Examples:
            return {"message": "success", "data": {alpr entitties}}

    * **Error Response**
        ::

            HTTP/1.1 401 UnAuthorized
            Status: 401 UnAuthorized
            Content-Type: application/json
            Reasons:
            Provided access token is not valid.
    """
    if not is_authenticated:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not check_resource_permissions(request=request):
        raise HTTPException(
            status_code=403,
            detail="You don't have enough permissions to access this resource.",
        )
    registered_plates = alpr_entities_col.find_one(
        {}, {"_id": 0}, sort=[("_id", pymongo.DESCENDING)]
    )
    if not registered_plates:
        alpr_entities = populate_alprEntities(default_known_stickers)
        sanitize(alpr_entities["known_entities"])
        alpr_entities_col.insert_one(alpr_entities)
        registered_plates = alpr_entities

    return {"message": "success", "data": registered_plates["known_entities"]}


@app.put("/alpr/entities", name="Put Entities")
async def update_alpr_entities(
        request: Request,
        alpr_entities: AlprEntities,
        is_authenticated: bool = Depends(authentication),
):
    """
    Api to update known license plates.

    * **URL**
        `localhost:8000/alpr/entities`

    * **Method**
        `[PUT]`

    * **Headers**
        * `'accept: application/json'`
        * `Authorization': 'bearer {token}'`

    * **Request Body**
        * `Entities (Required)`

            `dict`

        ::

            Example
            {
                "known_entities": [
                    {
                    "object_id": "string",
                    "object_type": "string",
                    "serial_number": "string"
                    }
                ]
            }

    * **Response**
        ::

            HTTP/1.1 200 OK
            Date: Thu, 24 Feb 2011 12:36:30 GMT
            Status: 200 OK
            Content-Type: application/json
            Examples
            {"message": "Alpr Entities updated successfully", "data": {db_id}}

    * **Error Response**
        ::

            HTTP/1.1 401 UnAuthorized
            Status: 401 UnAuthorized
            Content-Type: application/json
            Reasons:
            Provided access token is not valid.

    * **Error Response**
        ::

            HTTP/1.1 422 Validation Error
            Status: 422 Validation Error
            Content-Type: application/json
            Example:
            { "detail": [ { "loc": [ "string" ], "msg": "string", "type": "string" } ] }
    """
    if not is_authenticated:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not check_resource_permissions(request=request):
        raise HTTPException(
            status_code=403,
            detail="You don't have enough permissions to access this resource.",
        )
    alpr_entities = alpr_entities.dict()

    sanitize(alpr_entities["known_entities"])
    obj_id = alpr_entities_col.insert_one(alpr_entities).inserted_id
    return {"message": "Alpr Entities updated successfully", "data": str(obj_id)}


@app.post("/floor-plans/", name="Add Floor Plans")
def create_file(
        request: Request,
        sitePlan: UploadFile = File(...),
        is_authenticated: bool = Depends(authentication),
):
    """
    This api is responsible to add floor plan image into resource
    folder as well as metadata into the database with base64 encoding.

    * **URL**
        `localhost:8000/floor-plans/`


    * **Method**
        `[POST]`

    * **Headers**
        * `'accept: application/json'`
        * `Authorization': 'bearer {token}'`

    * **Request Body**
        * `sitePlan  (Required)`

            `string(binary) Image`

    * **Response**
        ::

            HTTP/1.1 200 OK
            Date: Thu, 24 Feb 2011 12:36:30 GMT
            Status: 200 OK
            Content-Type: application/json
            Examples
            {"message": "Floorplan added successfully", "data": {db_id}}

    * **Error Response**
        ::

            HTTP/1.1 401 UnAuthorized
            Status: 401 UnAuthorized
            Content-Type: application/json
            Reasons:
            Provided access token is not valid.

    * **Error Response**
        ::

            HTTP/1.1 422 Validation Error
            Status: 422 Validation Error
            Content-Type: application/json
            Example:
            { "detail": [ { "loc": [ "string" ], "msg": "string", "type": "string" } ] }
    """
    if not is_authenticated:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not check_resource_permissions(request=request):
        raise HTTPException(
            status_code=403,
            detail="You don't have enough permissions to access this resource.",
        )
    sitePlan = base64.b64encode(sitePlan.file.read())
    floor_plan_id = planCol.insert_one(
        {"created_at": datetime.datetime.now(), "sitePlan": str(sitePlan)}
    ).inserted_id
    project_settings = list(proj_settings.find())
    new_floorplan = {"$set": {"sitePlan": str(floor_plan_id)}}
    proj_settings.update_many(filter=project_settings, update=new_floorplan)
    return {"message": "Floorplan added successfully", "data": str(floor_plan_id)}


@app.get("/cam-test-connection", name="Test Camera")
def test_camera(
        request: Request,
        url: str,
        user: str = None,
        password: str = None,
        protocol: str = None,
        is_authenticated: bool = Depends(authentication),
):
    """
    This api is responsible to test camera connection, returns cam preview.

    * **URL**
        `localhost:8000/cam-test-connection`

    * **Method**
        `[GET]`

    * **Headers**
        * `'accept: application/json'`
        * `Authorization': 'bearer {token}'`

    * **URL Params**
        * `url (Required)`

            `String (RTSP cam URL)`

        * `user`

            `String (Credential Username)`

        * `password`

            `String (Credential Password)`

        * `protocol`

            `String`

    * **Response**
        ::

            HTTP/1.1 200 OK
            Date: Thu, 24 Feb 2011 12:36:30 GMT
            Status: 200 OK
            Content-Type: application/json
            Examples:
            {
                "message": "success",
                "image": frame,
                "width": image_size_width,
                "height": image_size_height,
                "fps": fps,
            }

    * **Error Response**
        ::

            HTTP/1.1 401 UnAuthorized
            Status: 401 UnAuthorized
            Content-Type: application/json
            Reasons:
            Provided access token is not valid.

    * **Error Response**
        ::

            HTTP/1.1 400 Bad Request
            Status: 400 Bad Request
            Content-Type: application/json
            Reasons:
            Invalid Request Parameters.

    * **Error Response**
        ::

            HTTP/1.1 404 Camera not Found
            Status: 404
            Content-Type: application/json
            Reasons:
            Camera is not accessibale or not available.

    * **Error Response**
        ::

            HTTP/1.1 408 Request timeout
            Status: 401
            Content-Type: application/json
            Reasons:
            Camera Response taking too much time.
    """
    if not is_authenticated:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not check_resource_permissions(request=request):
        raise HTTPException(
            status_code=403,
            detail="You don't have enough permissions to access this resource.",
        )
    url = sanitize(url)
    user = sanitize(user)
    password = sanitize(password)
    protocol = sanitize(protocol)

    is_valid, error = validate_cam_params(user, password, url)
    if not is_valid:
        raise HTTPException(status_code=400, detail="Bad request")

    if ENABLE_IMAGE_PREVIEW_TIMEOUT:
        return_value = Queue()
        process = Process(
            target=get_camera_preview,
            args=(
                url,
                user,
                password,
                protocol,
                return_value,
            ),
        )
        process.start()
        for i in range(IMAGE_PREVIEW_TIMEOUT):
            process.join(timeout=1)
            if not return_value.empty():
                value = return_value.get()
                if process.is_alive():
                    process.terminate()
                if isinstance(value, str):
                    raise HTTPException(status_code=404, detail=value)
                return value

        if process.is_alive():
            process.terminate()
            raise HTTPException(status_code=408, detail="Timeout")
    else:
        return get_camera_preview(url, user, password, protocol)


@app.post("/camera", name="Add Camera")
async def add_camera(
        request: Request, camera: Camera, is_authenticated: bool = Depends(authentication)
):
    """
    Create/on-board a new camera

    * **URL**
        `localhost:8000/camera`

    * **Method**
        `[POST]`

    * **Headers**
        * `'accept: application/json'`
        * `Authorization': 'bearer {token}'`

    * **Request Body**
        * `Camera Details  (Required)`

            `dict`

        ::

            Example
            {
                "cameraId": "string",
                "cameraEnabled": "boolean",
                "displayName": "string",
                "url": "string",
                "user": "string",
                "password": "string",
                "protocol"" "string",
                "homography": false,
                "src_points": [
                    "string"
                ],
                "dst_points": [
                    "string"
                ],
                "frameWidth": "int",
                "frameHeight": "int",
                "frames_per_second": "int",
                "createdAt": "string",
                "image": "string"
            }

    * **Response**
        ::

            HTTP/1.1 200 OK
            Date: Thu, 24 Feb 2011 12:36:30 GMT
            Status: 200 OK
            Content-Type: application/json
            Examples
            {"message": "Camera Added successfully", "data": {db_id}}

    * **Error Response**
        ::

            HTTP/1.1 401 UnAuthorized
            Status: 401 UnAuthorized
            Content-Type: application/json
            Reasons:
            Provided access token is not valid.

    * **Error Response**
        ::

            HTTP/1.1 422 Validation Error
            Status: 422 Validation Error
            Content-Type: application/json
            Example:
            { "detail": [ { "loc": [ "string" ], "msg": "string", "type": "string" } ] }
    """
    if not is_authenticated:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not check_resource_permissions(request=request):
        raise HTTPException(
            status_code=403,
            detail="You don't have enough permissions to access this resource.",
        )
    camera = camera.dict()
    camera["displayName"] = sanitize(camera["displayName"])
    camera["user"] = sanitize(camera["user"])
    camera["url"] = sanitize(camera["url"])
    camera["password"] = sanitize(camera["password"])
    camera["protocol"] = sanitize(camera["protocol"])

    is_valid, error = validate_cam_params(
        camera["user"], camera["password"], camera["url"], camera["displayName"]
    )
    if not is_valid:
        return {"Error": error}

    camera["cameraId"] = str(uuid.uuid4())
    camera["createdAt"] = datetime.datetime.now()
    sensors.insert_one(camera).inserted_id

    return {"message": "Camera Added successfully", "data": camera["cameraId"]}


@app.put("/camera", name="Update Camera")
async def update_camera(
        request: Request, camera: Camera, is_authenticated: bool = Depends(authentication)
):
    """
    Create/on-board a new camera

    * **URL**
        `localhost:8000/camera`

    * **Method**
        `[PUT]`

    * **Headers**
        * `'accept: application/json'`
        * `Authorization': 'bearer {token}'`

    * **Request Body**
        * `Camera Details  (Required)`

            `dict`

        ::

            Example
            {
                "cameraId": "string",
                "cameraEnabled": "boolean",
                "displayName": "string",
                "url": "string",
                "user": "string",
                "password": "string",
                "protocol"" "string",
                "homography": false,
                "src_points": [
                    "string"
                ],
                "dst_points": [
                    "string"
                ],
                "frameWidth": "int",
                "frameHeight": "int",
                "frames_per_second": "int",
                "createdAt": "string",
                "image": "string"
            }

    * **Response**
        ::

            HTTP/1.1 200 OK
            Date: Thu, 24 Feb 2011 12:36:30 GMT
            Status: 200 OK
            Content-Type: application/json
            Examples
            {"message": "Camera updated successfully"}

    * **Error Response**
        ::

            HTTP/1.1 401 UnAuthorized
            Status: 401 UnAuthorized
            Content-Type: application/json
            Reasons:
            Provided access token is not valid.

    * **Error Response**
        ::

            HTTP/1.1 404 Camera Not Found
            Status: 404 Camera Not Found
            Content-Type: application/json
            Reasons:
            Camera not found in database.

    * **Error Response**
        ::

            HTTP/1.1 422 Validation Error
            Status: 422 Validation Error
            Content-Type: application/json
            Example:
            { "detail": [ { "loc": [ "string" ], "msg": "string", "type": "string" } ] }
    """
    if not is_authenticated:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not check_resource_permissions(request=request):
        raise HTTPException(
            status_code=403,
            detail="You don't have enough permissions to access this resource.",
        )
    camera = camera.dict()
    camera["displayName"] = sanitize(camera["displayName"])
    camera["user"] = sanitize(camera["user"])
    camera["url"] = sanitize(camera["url"])
    camera["password"] = sanitize(camera["password"])
    camera["protocol"] = sanitize(camera["protocol"])

    is_valid, error = validate_cam_params(
        camera["user"], camera["password"], camera["url"], camera["displayName"]
    )
    if not is_valid:
        return {"Error": error}

    result = sensors.update_one({"cameraId": camera["cameraId"]}, {"$set": camera})

    if result.modified_count:
        return {"message": "Camera updated successfully"}
    else:
        raise HTTPException(status_code=404, detail="Camera not found")


@app.delete("/camera", name="Delete Cam")
async def delete_camera(
        request: Request, cameraId: str, is_authenticated: bool = Depends(authentication)
):
    """
    Delete specific camera providing camera id

    * **URL**
        `localhost:8000/camera`

    * **Method**
        `[DELETE]`

    * **Headers**
        * `'accept: application/json'`
        * `Authorization': 'bearer {token}'`

    * **Request Body**
        * `Camera ID (Required)`

            `string`

    * **Response**
        ::

            HTTP/1.1 200 OK
            Date: Thu, 24 Feb 2011 12:36:30 GMT
            Status: 200 OK
            Content-Type: application/json
            Examples
            {"message": "Camera deleted successfully"}

    * **Error Response**
        ::

            HTTP/1.1 401 UnAuthorized
            Status: 401 UnAuthorized
            Content-Type: application/json
            Reasons:
            Provided access token is not valid.

    * **Error Response**
        ::

            HTTP/1.1 404 Camera Not Found
            Status: 404 Camera Not Found
            Content-Type: application/json
            Reasons:
            Camera not found in database.

    * **Error Response**
        ::

            HTTP/1.1 422 Validation Error
            Status: 422 Validation Error
            Content-Type: application/json
            Example:
            { "detail": [ { "loc": [ "string" ], "msg": "string", "type": "string" } ] }
    """
    if not is_authenticated:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not check_resource_permissions(request=request):
        raise HTTPException(
            status_code=403,
            detail="You don't have enough permissions to access this resource.",
        )
    cameraId = sanitize(cameraId)
    result = sensors.delete_one({"cameraId": cameraId})

    if result.deleted_count:
        return {"message": "Camera deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="Camera not found")


@app.get("/cameras-list", name="Camera List")
async def camera_list(
        request: Request,
        pagination: Pagination = Depends(),
        camera_enable: bool = None,
        is_authenticated: bool = Depends(authentication),
):
    """
    Returns Paginated List of Cameras.

    * **URL**
        `localhost:8000/cameras-list`


    * **Method**
        `[GET]`

    * **Headers**
        * `'accept: application/json'`
        * `Authorization': 'bearer {token}'`

    * **URL Params**
        * `camera_enable`

            `BOOL`

        * `offset`

            `INT`

        * `limit`

            `INT`

    * **Response**
        ::

            HTTP/1.1 200 OK
            Date: Thu, 24 Feb 2011 12:36:30 GMT
            Status: 200 OK
            Content-Type: application/json
            Examples
            {
                "count": "count of items",
                "next": "next url",
                "previous": "previous url",
                "result": "items",
            }

    * **Error Response**
        ::

            HTTP/1.1 401 UnAuthorized
            Status: 401 UnAuthorized
            Content-Type: application/json
            Reasons:
            Provided access token is not valid.
    """
    if not is_authenticated:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not check_resource_permissions(request=request):
        raise HTTPException(
            status_code=403,
            detail="You don't have enough permissions to access this resource.",
        )
    response = await pagination.paginate(
        collection=sensors, camera_enable=camera_enable
    )
    for camera in response["result"]:
        camera["image"] = None
    return response


@app.get("/camera-preview", name="Camera Saved Preview")
async def get_camera_saved_preview(
        request: Request, camera_id: str, is_authenticated: bool = Depends(authentication)
):
    """
    Returns Paginated List of Cameras.

    * **URL**
        `localhost:8000/camera-preview`


    * **Method**
        `[GET]`

    * **Headers**
        * `'accept: application/json'`
        * `Authorization': 'bearer {token}'`

    * **URL Params**
        * `camera_id`

            `str`

    * **Response**
        ::

            HTTP/1.1 200 OK
            Date: Thu, 24 Feb 2011 12:36:30 GMT
            Status: 200 OK
            Content-Type: application/json
            Examples
            {
                "message": "success",
                "data": "Base 64 Image",
                "previous": "previous url",
                "result": "items",
            }

    * **Error Response**
        ::

            HTTP/1.1 401 UnAuthorized
            Status: 401 UnAuthorized
            Content-Type: application/json
            Reasons:
            Provided access token is not valid.

    * **Error Response**
        ::

            HTTP/1.1 404 UnAuthorized
            Status: 404 UnAuthorized
            Content-Type: application/json
            Reasons:
            Image Preview Not Found

    * **Error Response**
        ::

            HTTP/1.1 200 UnAuthorized
            Status: 200 UnAuthorized
            Content-Type: application/json
            Example
            {"Failure": f"Exception Occured {exception message}"}
            Reasons:
            Database Read Issues
    """
    if not is_authenticated:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not check_resource_permissions(request=request):
        raise HTTPException(
            status_code=403,
            detail="You don't have enough permissions to access this resource.",
        )
    camera_id = sanitize(camera_id)
    try:
        camera_preview = list(sensors.find({"cameraId": camera_id}, {"_id": 0}))
        if camera_preview:
            return {"message": "success", "data": camera_preview[0]["image"]}
    except Exception as e:
        return {"Failure": f"Exception Occured {e}"}

    raise HTTPException(status_code=404, detail="Camera Preview not found")


@app.get("/cameras", name="Cameras")
async def get_all_cameras(
        request: Request,
        camera_enable: bool = None,
        is_authenticated: bool = Depends(authentication),
):
    """
    Returns all the Cameras.

    * **URL**
        `localhost:8000/cameras`

    * **Method**
        `[GET]`

    * **Headers**
        * `'accept: application/json'`
        * `Authorization': 'bearer {token}'`

    * **URL Params**
        * `camera_enable`

            `BOOL`

    * **Response**
        ::

            HTTP/1.1 200 OK
            Date: Thu, 24 Feb 2011 12:36:30 GMT
            Status: 200 OK
            Content-Type: application/json
            Examples
            {"message": "success", "data": cameras}

    * **Error Response**
        ::

            HTTP/1.1 401 UnAuthorized
            Status: 401 UnAuthorized
            Content-Type: application/json
            Reasons:
            Provided access token is not valid.

    * **Error Response**
        ::

            HTTP/1.1 404 Cameras not Found
            Status: 404 Cameras not Found
            Content-Type: application/json
            Reasons:
            No cameras exist.
    """
    if not is_authenticated:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not check_resource_permissions(request=request):
        raise HTTPException(
            status_code=403,
            detail="You don't have enough permissions to access this resource.",
        )
    if not camera_enable:
        cameras = list(sensors.find({}, {"_id": 0}))
    else:
        cameras = list(sensors.find({"cameraEnabled": camera_enable}, {"_id": 0}))

    if cameras:
        return {"message": "success", "data": cameras}
    else:
        raise HTTPException(status_code=404, detail="cameras not found")


@app.post("/camera-mask", name="Add Camera Mask")
def add_camera_mask(
        request: Request,
        cameraId: str = Form(...),
        mask_points: str = Form(...),
        obscure_mask: UploadFile = File(...),
        is_authenticated: bool = Depends(authentication),
):
    """
    Add Camera Mask.

    * **URL**
        `localhost:8000/camera-mask`

    * **Method**
        `[POST]`

    * **Headers**
        * `'accept: application/json'`
        * `Authorization': 'bearer {token}'`

    * **Request Body**
        * `cameraId (Required)`

            `string`

        * `mask_points (Required)`

            `string`

        * `obscure_mask (Required)`

            `string(binary) Image`

    * **Response**
        ::

            HTTP/1.1 200 OK
            Date: Thu, 24 Feb 2011 12:36:30 GMT
            Status: 200 OK
            Content-Type: application/json
            Examples
            {"message": "Mask added successfully", "data":obscure_mask_id}

    * **Error Response**
        ::

            HTTP/1.1 401 UnAuthorized
            Status: 401 UnAuthorized
            Content-Type: application/json
            Reasons:
            Provided access token is not valid.

    * **Error Response**
        ::

            HTTP/1.1 422 Validation Error
            Status: 422 Validation Error
            Content-Type: application/json
            Example:
            { "detail": [ { "loc": [ "string" ], "msg": "string", "type": "string" } ] }
    """
    if not is_authenticated:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not check_resource_permissions(request=request):
        raise HTTPException(
            status_code=403,
            detail="You don't have enough permissions to access this resource.",
        )
    cameraId = sanitize(cameraId)
    mask_points = sanitize(mask_points, 10000)
    obscure_mask = base64.b64encode(obscure_mask.file.read())
    obscure_mask_id = obscure_mask_col.insert_one(
        {
            "created_at": datetime.datetime.now(),
            "obscure_mask": obscure_mask,
            "cameraId": cameraId,
            "mask_points": mask_points,
        }
    ).inserted_id
    return {"message": "Mask added successfully", "data": str(obscure_mask_id)}


@app.get("/camera-mask", name="Get Camera Mask")
async def get_camera_mask(
        request: Request,
        cameraId: str,
        is_authenticated: bool = Depends(authentication),
):
    """
    Get camera masks against camera id parameter

    * **URL**
        `localhost:8000/camera-mask`

    * **Method**
        `[GET]`

    * **Headers**
        * `'accept: application/json'`
        * `Authorization': 'bearer {token}'`

    * **URL Params**
        * `camera_id`

            `String`

    * **Response**
        ::

            HTTP/1.1 200 OK
            Date: Thu, 24 Feb 2011 12:36:30 GMT
            Status: 200 OK
            Content-Type: application/json
            Examples:
            {"message": "success", "data": camera_masks}

    * **Error Response**
        ::

            HTTP/1.1 401 UnAuthorized
            Status: 401 UnAuthorized
            Content-Type: application/json
            Reasons:
            Provided access token is not valid.

    * **Error Response**
        ::

            HTTP/1.1 404 Camera not Found
            Status: 404 Camera not Found
            Content-Type: application/json
            Reasons:
            Camera not found.
    """
    if not is_authenticated:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not check_resource_permissions(request=request):
        raise HTTPException(
            status_code=403,
            detail="You don't have enough permissions to access this resource.",
        )
    cameraId = sanitize(cameraId)
    cameraId = [x.strip() for x in cameraId.split(",")]
    obscure_masks = obscure_mask_col.find(
        {"cameraId": {"$in": cameraId}}, {"_id": 0, "created_at": 0}
    )
    if obscure_masks:
        camera_masks = collections.defaultdict(dict)
        for doc in obscure_masks:
            camera_masks[doc["cameraId"]]["obscure_mask"] = doc["obscure_mask"]
            camera_masks[doc["cameraId"]]["mask_points"] = doc["mask_points"]
        return {"message": "success", "data": camera_masks}
    else:
        raise HTTPException(status_code=404, detail="Camera not found")


@app.put("/container-state", name="Update Container State")
async def update_container_state(
        request: Request,
        service: ServiceContainer,
        is_authenticated: bool = Depends(authentication),
):
    """
    Update Container State.

    * **URL**
        `localhost:8000/container-state`


    * **Method**
        `[PUT]`

    * **Headers**
        * `'accept: application/json'`
        * `Authorization': 'bearer {token}'`

    * **Request Body**
        * `configs (Required)`

            `dict`

        ::

            Example
            {
            "container_name": "string",
            "state": "string"
            }

    * **Response**
        ::

            HTTP/1.1 200 OK
            Date: Thu, 24 Feb 2011 12:36:30 GMT
            Status: 200 OK
            Content-Type: application/json
            Examples
            {"message": "{container_name} state {state} changed successfully"}

    * **Error Response**
        ::

            HTTP/1.1 401 UnAuthorized
            Status: 401 UnAuthorized
            Content-Type: application/json
            Reasons:
            Provided access token is not valid.

    * **Error Response**
        ::

            HTTP/1.1 404 Not Found
            Status: 404 Not Found
            Content-Type: application/json
            Reasons:
            Incorrect state change.
            Container Not found.

    * **Error Response**
        ::

            HTTP/1.1 422 Validation Error
            Status: 422 Validation Error
            Content-Type: application/json
            Example:
            { "detail": [ { "loc": [ "string" ], "msg": "string", "type": "string" } ] }
    """
    if not is_authenticated:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not check_resource_permissions(request=request):
        raise HTTPException(
            status_code=403,
            detail="You don't have enough permissions to access this resource.",
        )
    client = docker.from_env()
    service = service.dict()
    if service.get("container_name") and service.get("state"):
        state = sanitize(service["state"])
        container_name = sanitize(service["container_name"])
        if client.containers.list(all=True, filters={"name": container_name}):
            status = client.containers.get(container_name).status
            if state == "pause" and status == "running":
                client.containers.get(container_name).pause()
            elif state == "unpause" and status == "paused":
                client.containers.get(container_name).unpause()
            elif state == "start" and status == "exited":
                client.containers.get(container_name).start()
            elif state == "stop" and status == "running":
                client.containers.get(container_name).stop()
            elif state == "reload":
                client.containers.get(container_name).reload()
            elif state == "restart":
                client.containers.get(container_name).restart()
            else:
                raise HTTPException(status_code=404, detail="Incorrect state change")
            return {"message": f"{container_name} state {state} changed successfully"}
        else:
            raise HTTPException(status_code=404, detail=f"{container_name} not found")
    else:
        raise HTTPException(status_code=404, detail="Key not found")


@app.get("/container-state", name="Get Container State")
async def get_container_state(
        request: Request,
        service_name: str = None,
        is_authenticated: bool = Depends(authentication),
):
    """
    Get status of specific or all docker containers

    * **URL**
        `localhost:8000/container-state`

    * **Method**
        `[GET]`

    * **Headers**
        * `'accept: application/json'`
        * `Authorization': 'bearer {token}'`

    * **URL Params**
        * `service_name`

            `String`

    * **Response**
        ::

            HTTP/1.1 200 OK
            Date: Thu, 24 Feb 2011 12:36:30 GMT
            Status: 200 OK
            Content-Type: application/json
            Examples
            {"message": "success", "data": Data}

    * **Error Response**
        ::

            HTTP/1.1 401 UnAuthorized
            Status: 401 UnAuthorized
            Content-Type: application/json
            Reasons:
            Provided access token is not valid.

    * **Error Response**
        ::

            HTTP/1.1 404 Service not Found
            Status: 404 Service not Found
            Content-Type: application/json
            Reasons:
            Service not found.
    """
    if not is_authenticated:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not check_resource_permissions(request=request):
        raise HTTPException(
            status_code=403,
            detail="You don't have enough permissions to access this resource.",
        )
    client = docker.from_env()
    if service_name:
        service_name = sanitize(service_name)
        if client.containers.list(all=True, filters={"name": service_name}):
            status = client.containers.get(service_name).status
            return {"message": "success", "data": status}
        else:
            raise HTTPException(status_code=404, detail=f"{service_name} not found")
    else:
        services = {}
        data = yaml.load(open("docker-compose.yml", "r"))
        containers = list(data["services"].keys())
        for container in client.containers.list(all=True):
            if container.name in containers:
                services[container.name] = container.status
            else:
                continue
        return {"message": "success", "data": services}


@app.get("/get-containers", name="Get All Containers Status")
async def get_containers(
        request: Request, is_authenticated: bool = Depends(authentication)
):
    """
    Get status of all docker containers

    * **URL**
        `localhost:8000/get-containers`

    * **Method**
        `[GET]`

    * **Headers**
        * `'accept: application/json'`
        * `Authorization': 'bearer {token}'`

    * **Response**
        ::

            HTTP/1.1 200 OK
            Date: Thu, 24 Feb 2011 12:36:30 GMT
            Status: 200 OK
            Content-Type: application/json
            Examples
            {"code": 200, "message": "success", "data": services}

    * **Error Response**
        ::

            HTTP/1.1 401 UnAuthorized
            Status: 401 UnAuthorized
            Content-Type: application/json
            Reasons:
            Provided access token is not valid.
    """
    if not is_authenticated:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not check_resource_permissions(request=request):
        raise HTTPException(
            status_code=403,
            detail="You don't have enough permissions to access this resource.",
        )
    client = docker.from_env()
    services = {}
    data = yaml.load(open("docker-compose.yml", "r"))
    containers = list(data["services"].keys())
    for container in client.containers.list(all=True):
        if container.name in containers:
            services[container.name] = container.status
        else:
            continue
    return {"code": 200, "message": "success", "data": services}


def add_obj_type(obj):
    """
    Adds object types
    """
    if isinstance(obj, dict):
        new_dict = {}
        for key, value in obj.items():
            if key.lower().find("host") != -1 or key.count(".") == 3:
                new_dict[key] = [value, "ip"]
            else:
                new_dict[key] = add_obj_type(value)
        return new_dict
    else:
        return [obj, type(obj).__name__]


@app.get("/services-config-with-type", name="Service Configs with types")
async def get_services_config_with_type(
        request: Request, is_authenticated: bool = Depends(authentication)
):
    """
    Get all services configs with item type.

    * **URL**
        `localhost:8000/services-config-with-type`

    * **Method**
        `[GET]`

    * **Headers**
        * `'accept: application/json'`
        * `Authorization': 'bearer {token}'`

    * **Response**
        ::

            HTTP/1.1 200 OK
            Date: Thu, 24 Feb 2011 12:36:30 GMT
            Status: 200 OK
            Content-Type: application/json
            Examples
            {"message": "success", "data": Data}

    * **Error Response**
        ::

            HTTP/1.1 401 UnAuthorized
            Status: 401 UnAuthorized
            Content-Type: application/json
            Reasons:
            Provided access token is not valid.
    """
    if not is_authenticated:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not check_resource_permissions(request=request):
        raise HTTPException(
            status_code=403,
            detail="You don't have enough permissions to access this resource.",
        )
    configs = services_settings.find_one(
        {}, {"_id": 0}, sort=[("_id", pymongo.DESCENDING)]
    )
    if configs:
        configs_with_data_types = add_obj_type(configs)
        categorized_configs = categorised_config_data(configs=configs_with_data_types)
        return {"code": 200, "message": "success", "data": categorized_configs}
    else:
        raise HTTPException(status_code=404, detail="settings not found")


@app.get("/services-config", name="Service Configs")
async def get_services_config(
        request: Request,
        service: str = None,
        is_authenticated: bool = Depends(authentication),
):
    """
    Get all services configs.

    * **URL**
        `localhost:8000/services-config`

    * **Method**
        `[GET]`

    * **Headers**
        * `'accept: application/json'`
        * `Authorization': 'bearer {token}'`

    * **URL Params**
        * `service_name`

            `String`

    * **Response**
        ::

            HTTP/1.1 200 OK
            Date: Thu, 24 Feb 2011 12:36:30 GMT
            Status: 200 OK
            Content-Type: application/json
            Examples:
            {"message": "success", "data": Data}

    * **Error Response**
        ::

            HTTP/1.1 401 UnAuthorized
            Status: 401 UnAuthorized
            Content-Type: application/json
            Reasons:
            Provided access token is not valid.

    * **Error Response**
        ::

            HTTP/1.1 404 Service not Found
            Status: 404 Service not Found
            Content-Type: application/json
            Reasons:
            Service not found.
    """
    if not is_authenticated:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not check_resource_permissions(request=request):
        raise HTTPException(
            status_code=403,
            detail="You don't have enough permissions to access this resource.",
        )
    service = sanitize(service)
    configs = services_settings.find_one(
        {}, {"_id": 0}, sort=[("_id", pymongo.DESCENDING)]
    )
    if configs:
        if service and configs.get(service):
            configs = configs[service]
        elif service and not configs.get(service):
            raise HTTPException(status_code=404, detail=f"{service} not found")
        return {"message": "success", "data": configs}
    else:
        raise HTTPException(status_code=404, detail="settings not found")


@app.post("/services-config", name="Update Services Configs")
async def add_services_config(
        request: Request,
        configs: ServicesConfigs,
        is_authenticated: bool = Depends(authentication),
):
    """
    Add services configs.

    * **URL**
        `localhost:8000/services-config`

    * **Method**
        `[POST]`

    * **Headers**
        * `'accept: application/json'`
        * `Authorization': 'bearer {token}'`

    * **Request Body**
        * `configs (Required)`

            `dict`

        ::

            Example:
            {
                "ingestor": {},
                "smoother": {},
                "streamer": {},
                "redis_graylog": {},
                "cleaner": {},
                "transmitter": {},
                "alpr": {},
                "historian": {},
                "enricher": {}
            }

    * **Response**
        ::

            HTTP/1.1 200 OK
            Date: Thu, 24 Feb 2011 12:36:30 GMT
            Status: 200 OK
            Content-Type: application/json
            Examples:
             {"message": "Settings added successfully", "ID": db_id)}

    * **Error Response**
        ::

            HTTP/1.1 401 UnAuthorized
            Status: 401 UnAuthorized
            Content-Type: application/json
            Reasons:
            Provided access token is not valid.

    * **Error Response**
        ::

            HTTP/1.1 422 Validation Error
            Status: 422 Validation Error
            Content-Type: application/json
            Example:
            { "detail": [ { "loc": [ "string" ], "msg": "string", "type": "string" } ] }
    """
    if not is_authenticated:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not check_resource_permissions(request=request):
        raise HTTPException(
            status_code=403,
            detail="You don't have enough permissions to access this resource.",
        )
    settings = services_settings.find_one(
        {}, {"_id": 0}, sort=[("_id", pymongo.DESCENDING)]
    )
    configs = configs.dict()

    if not settings:
        settings = {}
    else:  # Handle deleted Keys
        for key in list(settings.keys()):
            if settings[key]:
                for conf in list(settings[key].keys()):
                    if configs[key] and not configs[key].get(conf):
                        settings[key].pop(conf)

    # Add and Modify Keys
    for key in configs:
        if not configs[key]:
            settings[key] = None
        elif not settings.get(key):
            settings[key] = {}
            settings[key] = configs[key]
            for conf in configs[key]:
                settings[key][conf] = get_value_from_data(configs[key][conf])
        else:
            for conf in configs[key]:
                if not settings[key].get(conf):
                    settings[key][conf] = {}
                settings[key][conf] = get_value_from_data(configs[key][conf])
    col_id = services_settings.insert_one(settings).inserted_id
    return {"message": "Settings added successfully", "ID": str(col_id)}


@app.put("/services-config", name="Update Services Configs")
async def update_services_config(
        request: Request, configs: dict, is_authenticated: bool = Depends(authentication)
):
    """
    Update services configs.

    * **URL**
        `localhost:8000/services-config`

    * **Method**
        `[PUT]`

    * **Headers**
        * `'accept: application/json'`
        * `Authorization': 'bearer {token}'`

    * **Request Body**
        * `configs (Required)`

            `dict`

        ::

            Example:
            {
                "ingestor": {},
                "smoother": {},
                "streamer": {},
                "redis_graylog": {},
                "cleaner": {},
                "transmitter": {},
                "alpr": {},
                "historian": {},
                "enricher": {}
            }

    * **Response**
        ::

            HTTP/1.1 200 OK
            Date: Thu, 24 Feb 2011 12:36:30 GMT
            Status: 200 OK
            Content-Type: application/json
            Examples:
             {"message": "Settings added successfully", "ID": db_id)}

    * **Error Response**
        ::

            HTTP/1.1 401 UnAuthorized
            Status: 401 UnAuthorized
            Content-Type: application/json
            Reasons:
            Provided access token is not valid.

    * **Error Response**
        ::

            HTTP/1.1 422 Validation Error
            Status: 422 Validation Error
            Content-Type: application/json
            Example:
            { "detail": [ { "loc": [ "string" ], "msg": "string", "type": "string" } ] }
    """
    if not is_authenticated:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not check_resource_permissions(request=request):
        raise HTTPException(
            status_code=403,
            detail="You don't have enough permissions to access this resource.",
        )
    settings = services_settings.find_one({}, sort=[("_id", pymongo.DESCENDING)])
    if not settings:
        settings = {}
    # Add and Modify Keys
    logging.info(f"db settings are {settings}")
    for key in configs:
        if not configs[key]:
            settings[key] = None
        elif not settings.get(key):
            settings[key] = {}
            settings[key] = configs[key]
            for conf in configs[key]:
                settings[key][conf] = get_value_from_data(configs[key][conf])
        else:
            for conf in configs[key]:
                if not settings[key].get(conf):
                    settings[key][conf] = {}
                settings[key][conf] = get_value_from_data(configs[key][conf])
    services_settings.replace_one(
        filter={"_id": ObjectId(str(settings.get("_id")))}, replacement=settings
    )
    return {
        "code": 200,
        "message": "Settings updated successfully",
        "data": {"id": str(settings.get("_id"))},
    }


@app.post("/backup-db-s3", name="Backup Database")
def backup_to_s3(
        request: Request,
        data: BucketUpload,
        is_authenticated: bool = Depends(authentication),
):
    """
    Create database backup from s3 Bucket.

    * **URL**
        `localhost:8000/restore-db-s3`

    * **Method**
        `[POST]`

    * **Headers**
        * `'accept: application/json'`
        * `Authorization': 'bearer {token}'`

    * **URL Params**
        * `data`

            `String`

    * **Response**
        ::

            HTTP/1.1 200 OK
            Date: Thu, 24 Feb 2011 12:36:30 GMT
            Status: 200 OK
            Content-Type: application/json
            Examples
            {"message": "database has been backed up successfully as backup"}

    * **Error Response**
        ::

            HTTP/1.1 401 UnAuthorized
            Status: 401 UnAuthorized
            Content-Type: application/json
            Reasons:
            Provided access token is not valid.
    """
    if not is_authenticated:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not check_resource_permissions(request=request):
        raise HTTPException(
            status_code=403,
            detail="You don't have enough permissions to access this resource.",
        )
    data = data.dict()
    data["name"] = sanitize(data["name"])
    backup_name = database
    if data:
        backup_name = database + "_" + data["name"]
    command = f"mongodump --host {db_host} --port {db_port} --db {database}"
    command += " --forceTableScan --out /data"
    os.system(command)
    backup_cmd = "aws s3 sync "
    backup_cmd += f"/data/{database} "
    backup_cmd += "s3://${BUCKET}"
    backup_cmd += f"/{backup_name}"
    os.system(backup_cmd)
    return {"message": f"{database} has been backed up successfully as {backup_name}"}


@app.get("/restore-db-s3", name="Restore Backup")
def restore_from_s3(
        request: Request, is_authenticated: bool = Depends(authentication), name: str = None
):
    """
    Restore database from s3 Bucket

    * **URL**
        `localhost:8000/restore-db-s3`

    * **Method**
        `[GET]`

    * **Headers**
        * `'accept: application/json'`
        * `Authorization': 'bearer {token}'`

    * **URL Params**
        * `name`

            `String`

    * **Response**
        ::

            HTTP/1.1 200 OK
            Date: Thu, 24 Feb 2011 12:36:30 GMT
            Status: 200 OK
            Content-Type: application/json
            Examples
            {"message": "database has been backed up successfully as backup"}

    * **Error Response**
        ::

            HTTP/1.1 401 UnAuthorized
            Status: 401 UnAuthorized
            Content-Type: application/json
            Reasons:
            Provided access token is not valid.
    """
    if not is_authenticated:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not check_resource_permissions(request=request):
        raise HTTPException(
            status_code=403,
            detail="You don't have enough permissions to access this resource.",
        )
    name = sanitize(name)
    backup_name = database
    if name:
        backup_name = database + "_" + name
    restore_cmd = "aws s3 sync "
    restore_cmd += "s3://${BUCKET}"
    restore_cmd += f"/{backup_name} "
    restore_cmd += f"/data/{backup_name}"
    os.system(restore_cmd)
    command = f"mongorestore --host {db_host} --port {db_port} --db {database} --drop"
    command += f" /data/{backup_name}"
    os.system(command)
    return {"message": f"{database} has been backed up successfully as {backup_name}"}


@app.get("/list-backups", name="List Backups")
async def list_backups_s3(
        request: Request,
        is_authenticated: bool = Depends(authentication),
):
    """
    List backups present on s3 Bucket

    * **URL**
        `localhost:8000/list-backups`


    * **Method**
        `[GET]`

    * **Headers**
        * `'accept: application/json'`
        * `Authorization': 'bearer {token}'`

    * **Response**
        ::

            HTTP/1.1 200 OK
            Date: Thu, 24 Feb 2011 12:36:30 GMT
            Status: 200 OK
            Content-Type: application/json
            Examples
            {"message": "success", "data": backups}

    * **Error Response**
        ::

            HTTP/1.1 401 UnAuthorized
            Status: 401 UnAuthorized
            Content-Type: application/json
            Reasons:
            Provided access token is not valid.
    """
    if not is_authenticated:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not check_resource_permissions(request=request):
        raise HTTPException(
            status_code=403,
            detail="You don't have enough permissions to access this resource.",
        )
    command = "aws s3 ls 's3://darvis-config-backup/'"
    backups = os.popen(command).read()
    backups = backups.split()
    backups = [x.replace("/", "") for x in backups if x != "PRE"]
    return {"message": "success", "data": backups}


@app.post("/add-zone", name="Add Zone")
async def create_user_zone(
        request: Request,
        zones: str = Form(...),
        zone_id: str = Form(...),
        delete_zone_name: str = Form(...),
        edit_zone_name: str = Form(...),
        is_authenticated: bool = Depends(authentication),
):
    """
    Add New Zone.

    * **URL**
        `localhost:8000/edit-alert`

    * **Method**
        `[PUT]`

    * **Headers**
        * `'accept: application/json'`
        * `Authorization': 'bearer {token}'`

    * **Request Body**
        * `zones (Required)`

            `string`

        * `zone_id (Required)`

            `string`

        * `delete_zone_name (Required)`

            `string`

        * `edit_zone_name (Required)`

            `string`

    * **Response**
        ::

            HTTP/1.1 200 OK
            Date: Thu, 24 Feb 2011 12:36:30 GMT
            Status: 200 OK
            Content-Type: application/json
            Examples:
             {"code": 200, "message": "success", "data": {db_id}}

    * **Error Response**
        ::

            HTTP/1.1 401 UnAuthorized
            Status: 401 UnAuthorized
            Content-Type: application/json
            Reasons:
            Provided access token is not valid.

    * **Error Response**
        ::

            HTTP/1.1 404 Resource Not Found
            Status: 404 Resource Not Found
            Content-Type: application/json
            Reasons:
            When provided alert already exists.

    * **Error Response**
        ::

            HTTP/1.1 422 Validation Error
            Status: 422 Validation Error
            Content-Type: application/json
            Example:
            { "detail": [ { "loc": [ "string" ], "msg": "string", "type": "string" } ] }
    """
    if not is_authenticated:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not check_resource_permissions(request=request):
        raise HTTPException(
            status_code=403,
            detail="You don't have enough permissions to access this resource.",
        )
    data = {
        "zones": zones,
        "zone_id": zone_id,
        "delete_zone_name": delete_zone_name,
        "edit_zone_name": json.loads(edit_zone_name),
    }
    inserted_id = insert_zone(data=data, logging=logging)
    return {"code": 200, "message": "success", "data": {"zone_id": str(inserted_id)}}


@app.put("/edit-zone", name="Edit Zone")
async def edit_user_zone(
        request: Request,
        zone_name: str = Form(...),
        coordinates: str = Form(...),
        zone_id: str = Form(...),
        is_authenticated: bool = Depends(authentication),
):
    """
    Edit Zone.

    * **URL**
        `localhost:8000/edit-alert`

    * **Method**
        `[PUT]`

    * **Headers**
        * `'accept: application/json'`
        * `Authorization': 'bearer {token}'`

    * **Request Body**
        * `zone_name (Required)`

            `string`

        * `coordinates (Required)`

            `string`

        * `zone_id (Required)`

            `string`

    * **Response**
        ::

            HTTP/1.1 200 OK
            Date: Thu, 24 Feb 2011 12:36:30 GMT
            Status: 200 OK
            Content-Type: application/json
            Examples
             {"code": 200, "message": "success"}

    * **Error Response**
        ::

            HTTP/1.1 401 UnAuthorized
            Status: 401 UnAuthorized
            Content-Type: application/json
            Reasons:
            Provided access token is not valid.

    * **Error Response**
        ::

            HTTP/1.1 404 Resource Not Found
            Status: 404 Resource Not Found
            Content-Type: application/json
            Reasons:
            When provided alert already exists.

    * **Error Response**
        ::

            HTTP/1.1 422 Validation Error
            Status: 422 Validation Error
            Content-Type: application/json
            Example:
            { "detail": [ { "loc": [ "string" ], "msg": "string", "type": "string" } ] }
    """
    if not is_authenticated:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not check_resource_permissions(request=request):
        raise HTTPException(
            status_code=403,
            detail="You don't have enough permissions to access this resource.",
        )
    data = {"zone_name": zone_name, "coordinates": coordinates, "zone_id": zone_id}
    zones = edit_zone(data=data, logging=logging)
    if zones:
        return {"code": 200, "message": "success"}
    return HTTPException(status_code=404, detail="not found")


@app.get("/get-zone", name="Zones")
async def get_user_zone(
        request: Request, is_authenticated: bool = Depends(authentication)
):
    """
    Get user-define zones.

    * **URL**
        `localhost:8000/get-zones`

    * **Method**
        `[GET]`

    * **Headers**
        * `'accept: application/json'`
        * `Authorization': 'bearer {token}'`

    * **Response**
        ::

            HTTP/1.1 200 OK
            Date: Thu, 24 Feb 2011 12:36:30 GMT
            Status: 200 OK
            Content-Type: application/json
            Examples
            {"code": 200, "message": "success", "data": zones}

    * **Error Response**
        ::

            HTTP/1.1 401 UnAuthorized
            Status: 401 UnAuthorized
            Content-Type: application/json
            Reasons:
            Provided access token is not valid.

    * **Error Response**
        ::

            HTTP/1.1 404 Zones not Found
            Status: 404 Zones not Found
            Content-Type: application/json
            Reasons:
            No zones exist.
    """
    if not is_authenticated:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not check_resource_permissions(request=request):
        raise HTTPException(
            status_code=403,
            detail="You don't have enough permissions to access this resource.",
        )
    zones = get_zone(logging=logging)
    if zones:
        return {"code": 200, "message": "success", "data": zones}
    return {"code": 200, "message": "Zones not found", "data": None}


@app.delete("/delete-zone", name="Delete Zone")
async def delete_user_zone(
        request: Request,
        zone_id: str = None,
        is_authenticated: bool = Depends(authentication),
):
    """
    Delete specific camera providing camera id

    * **URL**
        `localhost:8000/delete_zone`


    * **Method**
        `[DELETE]`

    * **Headers**
        * `'accept: application/json'`
        * `Authorization': 'bearer {token}'`

    * **Request Body**
        * `Zone ID (Required)`

            `string`

    * **Response**
        ::

            HTTP/1.1 200 OK
            Date: Thu, 24 Feb 2011 12:36:30 GMT
            Status: 200 OK
            Content-Type: application/json
            Examples
            {"message": "success"}

    * **Error Response**
        ::

            HTTP/1.1 401 UnAuthorized
            Status: 401 UnAuthorized
            Content-Type: application/json
            Reasons:
            Provided access token is not valid.

    * **Error Response**
        ::

            HTTP/1.1 404 Zone Not Found
            Status: 404 Zone Not Found
            Content-Type: application/json
            Reasons:
            Camera not found in database.

    * **Error Response**
        ::

            HTTP/1.1 422 Validation Error
            Status: 422 Validation Error
            Content-Type: application/json
            Example:
            { "detail": [ { "loc": [ "string" ], "msg": "string", "type": "string" } ] }
    """
    if not is_authenticated:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not check_resource_permissions(request=request):
        raise HTTPException(
            status_code=403,
            detail="You don't have enough permissions to access this resource.",
        )
    zones = delete_zone(zone_id, logging)
    if zones:
        return {"code": 200, "message": "success"}
    return HTTPException(status_code=404, detail="Zone not found")


@app.post("/add-alert", name="Add Alert")
async def add_alert_configs(
        request: Request,
        name: str = Form(...),
        object_type: str = Form(default=None),
        dwell_time: int = Form(default=0),
        container_count: int = Form(default=0),
        occupancy_range: int = Form(default=0),
        sack_color: str = Form(default=None),
        time_slot_list: str = Form(default=[]),
        non_vehicle: bool = Form(default=True),
        is_enable: bool = Form(...),
        is_active: str = Form(...),
        is_absent: bool = Form(default=True),
        alert_type: int = Form(...),
        schedules: str = Form(...),
        is_authenticated: bool = Depends(authentication),
):
    """
    Add new Alert.

    * **URL**
        `localhost:8000/add-alert`

    * **Method**
        `[POST]`

    * **Headers**
        * `'accept: application/json'`
        * `Authorization': 'bearer {token}'`

    * **Request Body**
        * `name (Required)`

            `string`

        * `object_type (Optional)`

            `string`

        * `dwell_time (Optional)`

            `string`

        * `container_count (Optional)`

            `interger`

        * `occupancy_range (Optional)`

            `interger`

        * `sack_color (Optional)`

            `string`

        * `is_enable (Required)`

            `Boolean`

        * `is_active (Required)`

            `Boolean`

        * `is_absent (Optional)`

            `Boolean`

        * `alert_type (Required)`

            `interger`

        * `non_vehicle (Optional)`

            `Boolean`

        * `schedules (Required)`

            `string`

        * `time_slot_list (Required)`

            `string`

    * **Response**
        ::

            HTTP/1.1 200 OK
            Date: Thu, 24 Feb 2011 12:36:30 GMT
            Status: 200 OK
            Content-Type: application/json
            Examples
             {"code": 200, "message": "success", "data": db_data_id}

    * **Error Response**
        ::

            HTTP/1.1 400 Invalid schedules
            Status: 400 Invalid schedules
            Content-Type: application/json
            Reasons:
            Provided schedules are invalid.

    * **Error Response**
        ::

            HTTP/1.1 401 UnAuthorized
            Status: 401 UnAuthorized
            Content-Type: application/json
            Reasons:
            Provided access token is not valid.

    * **Error Response**
        ::

            HTTP/1.1 409 Resource already exist
            Status: 409 Resource already exist
            Content-Type: application/json
            Reasons:
            When provided alert already exists.

    * **Error Response**
        ::

            HTTP/1.1 422 Validation Error
            Status: 422 Validation Error
            Content-Type: application/json
            Example:
            { "detail": [ { "loc": [ "string" ], "msg": "string", "type": "string" } ] }
    """
    if not is_authenticated:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not check_resource_permissions(request=request):
        raise HTTPException(
            status_code=403,
            detail="You don't have enough permissions to access this resource.",
        )
    if validate_schedule(schedules=json.loads(schedules)):
        alert_config = {
            "name": name,
            "object_type": object_type,
            "is_enable": is_enable,
            "is_active": is_active,
            "alert_type": alert_type,
            "is_absent": is_absent,
            "schedules": json.loads(schedules),
            "time_slot_list": time_slot_list,
            "non_vehicle": non_vehicle,
            "sack_color": sack_color.lower().strip()
            if sack_color is not None
            else sack_color,
        }
        if AlertType(alert_type).name == "space":
            space_attributes = {
                "occupancy_range": occupancy_range,
                "container_count": container_count,
                "sack_color": sack_color.lower().strip()
                if sack_color is not None
                else sack_color,
            }
            alert_config.update(space_attributes)
        else:
            alert_config.update({"dwell_time": dwell_time})
        if is_absent:
            alert_config.update({"last_detection_time": time.time()})

        responce = add_alert(data=alert_config, logging=logging)
        if responce:
            return {
                "code": 200,
                "message": "success",
                "data": {"zone_id": str(responce)},
            }
        raise HTTPException(status_code=409, detail="Resource already exist")
    else:
        raise HTTPException(status_code=400, detail="Invalid schedules")


@app.put("/edit-alert", name="Edit Alert")
async def edit_alert_configs(
        request: Request,
        name: str = Form(...),
        object_type: str = Form(default=None),
        dwell_time: int = Form(default=0),
        container_count: int = Form(default=0),
        occupancy_range: int = Form(default=0),
        sack_color: str = Form(default=None),
        time_slot_list: str = Form(default=[]),
        non_vehicle: bool = Form(default=None),
        is_enable: bool = Form(...),
        alert_id: str = Form(...),
        is_active: str = Form(...),
        is_absent: bool = Form(default=True),
        alert_type: int = Form(...),
        schedules: str = Form(...),
        is_authenticated: bool = Depends(authentication),
):
    """
    Edit Alert.

    * **URL**
        `localhost:8000/edit-alert`

    * **Method**
        `[PUT]`

    * **Headers**
        * `'accept: application/json'`
        * `Authorization': 'bearer {token}'`

    * **Request Body**
        * `name (Required)`

            `string`

        * `object_type (Optional)`

            `string`

        * `dwell_time (Optional)`

            `string`

        * `container_count (Optional)`

            `interger`

        * `occupancy_range (Optional)`

            `interger`

        * `sack_color (Optional)`

            `string`

        * `is_enable (Required)`

            `Boolean`

        * `alert_id (Required)`

            `string`

        * `is_absent (Optional)`

            `Boolean`

        * `non_vehicle (Optional)`

            `string`

        * `is_active (Required)`

            `string`

        * `alert_type (Required)`

            `interger`

        * `schedules (Required)`

            `string`

        * `time_slot_list (Required)`

            `string`

    * **Response**
        ::

            HTTP/1.1 200 OK
            Date: Thu, 24 Feb 2011 12:36:30 GMT
            Status: 200 OK
            Content-Type: application/json
            Examples
            {"message": "success"}

    * **Error Response**
        ::

            HTTP/1.1 400 Bad request
            Status: 400 Bad request
            Content-Type: application/json
            Reasons:
            Invalid Schedules.

    * **Error Response**
        ::

            HTTP/1.1 401 UnAuthorized
            Status: 401 UnAuthorized
            Content-Type: application/json
            Reasons:
            Provided access token is not valid.

    * **Error Response**
        ::

            HTTP/1.1 409 Resource already exist
            Status: 409 Resource already exist
            Content-Type: application/json
            Reasons:
            When provided alert already exists.

    * **Error Response**
        ::

            HTTP/1.1 500 Server error
            Status: 500 Server error
            Content-Type: application/json
            Reasons:
            Unexpected behavior.

    * **Error Response**
        ::

            HTTP/1.1 422 Validation Error
            Status: 422 Validation Error
            Content-Type: application/json
            Example:
            { "detail": [ { "loc": [ "string" ], "msg": "string", "type": "string" } ] }
    """
    if not is_authenticated:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not check_resource_permissions(request=request):
        raise HTTPException(
            status_code=403,
            detail="You don't have enough permissions to access this resource.",
        )
    if validate_schedule(schedules=json.loads(schedules)):
        alert_config = {
            "name": name,
            "object_type": object_type,
            "is_enable": is_enable,
            "is_active": is_active,
            "alert_type": alert_type,
            "is_absent": is_absent,
            "schedules": json.loads(schedules),
            "time_slot_list": time_slot_list,
            "non_vehicle": non_vehicle,
            "sack_color": sack_color.lower().strip()
            if sack_color is not None
            else sack_color,
        }
        if AlertType(alert_type).name == "space":
            space_attributes = {
                "occupancy_range": occupancy_range,
                "container_count": container_count,
                "sack_color": sack_color.lower().strip()
                if sack_color is not None
                else sack_color,
            }
            alert_config.update(space_attributes)
        else:
            alert_config.update({"dwell_time": dwell_time})
        if is_absent:
            alert_config.update({"last_detection_time": time.time()})
        alert = edit_alert(data=alert_config, logging=logging, alert_id=alert_id)
        if alert == 409:
            raise HTTPException(status_code=409, detail="Resource already exist")
        elif alert:
            return {"code": 200, "message": "success", "data": None}
        else:
            raise HTTPException(status_code=500, detail="Internal server error")
    else:
        raise HTTPException(status_code=400, detail="Invalid schedules")


@app.get("/get-alert", name="Get Alerts")
async def get_zone_alert(
        request: Request,
        alert_id: str,
        is_authenticated: bool = Depends(authentication),
):
    """
    Get all the alerts

    * **URL**
        `localhost:8000/get-alert`

    * **Method**
        `[GET]`

    * **Headers**
        * `'accept: application/json'`
        * `Authorization': 'bearer {token}'`

    * **Request Body**
        * `Alert ID (Required)`

            `string`

    * **Response**
        ::

            HTTP/1.1 200 OK
            Date: Thu, 24 Feb 2011 12:36:30 GMT
            Status: 200 OK
            Content-Type: application/json
            Examples
            {"code": 200, "message": "success", "data": zone_alert}

    * **Error Response**
        ::

            HTTP/1.1 401 UnAuthorized
            Status: 401 UnAuthorized
            Content-Type: application/json
            Reasons:
            Provided access token is not valid.

    * **Error Response**
        ::

            HTTP/1.1 404 Alert not Found
            Status: 404 Alert not Found
            Content-Type: application/json
            Reasons:
            No Alert exist.
    """
    if not is_authenticated:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not check_resource_permissions(request=request):
        raise HTTPException(
            status_code=403,
            detail="You don't have enough permissions to access this resource.",
        )
    zone_alert = get_zone_alerts(alert_id=alert_id)
    if zone_alert:
        return {"code": 200, "message": "success", "data": zone_alert}
    return {"code": 404, "message": "Alert not found", "data": None}


@app.get("/alert-history", name="Alert History")
async def get_zone_notifications(
        request: Request,
        start_timestamp: int,
        end_timestamp: int,
        is_authenticated: bool = Depends(authentication),
):
    """
    Get user-define zones alert notifications

    * **URL**
        `localhost:8000/alert-history`

    * **Method**
        `[GET]`

    * **Headers**
        * `'accept: application/json'`
        * `Authorization': 'bearer {token}'`

    * **Request Body**
        * `start_timestamp (Required)`

            `int`

        * `end_timestamp (Required)`

            `int`

    * **Response**
        ::

            HTTP/1.1 200 OK
            Date: Thu, 24 Feb 2011 12:36:30 GMT
            Status: 200 OK
            Content-Type: application/json
            Examples
            {"code": 200, "message": "success", "data": notifications}

    * **Error Response**
        ::

            HTTP/1.1 401 UnAuthorized
            Status: 401 UnAuthorized
            Content-Type: application/json
            Reasons:
            Provided access token is not valid.

    * **Error Response**
        ::

            HTTP/1.1 404 Notification not Found
            Status: 404 Notification not Found
            Content-Type: application/json
            Reasons:
            No Alert exist.
    """
    if not is_authenticated:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not check_resource_permissions(request=request):
        raise HTTPException(
            status_code=403,
            detail="You don't have enough permissions to access this resource.",
        )
    if start_timestamp > end_timestamp:
        return {"code": 400, "message": "Invalid Params", "data": None}
    notifications = get_alerts_notifications(
        start_timestamp=start_timestamp, end_timestamp=end_timestamp
    )
    if notifications:
        return {"code": 200, "message": "success", "data": notifications}
    return {"code": 404, "message": "Notifications not found", "data": None}


@app.get("/open-alerts", name="Get open alerts")
async def get_open_alerts(
        request: Request, is_authenticated: bool = Depends(authentication)
):
    """
    Get all alerts that have open state.

    * **URL**
        `localhost:8000/open-alerts`

    * **Method**
        `[GET]`

    * **Headers**
        * `'accept: application/json'`
        * `Authorization': 'bearer {token}'`

    * **Response**
        ::

            HTTP/1.1 200 OK
            Date: Thu, 24 Feb 2011 12:36:30 GMT
            Status: 200 OK
            Content-Type: application/json
            Examples
            {"code": 200, "message": "success", "data": [
        {
            "id": "c1323f9a-607c-4887-8413-dfaa41f381f2",
            "name": "Zone A",
            "dwell_time": 119.0308849811554,
            "object_type": "universal_mail_container",
            "camera_id": "c211afbb-0608-4725-82c6-ffc540e29e02",
            "tmpfs_link": "/tmpfs/MAIN_1636457618848017000.jpg",
            "loc_x": 0.1496303230524063,
            "loc_y": 0.4158497750759125,
            "state": "open",
            "object_ids": [],
            "current_timestamp": 1636457619,
            "alert_type": "zone"
        },
        {
            "id": "b6fd7dd6-51f3-49da-a784-ebb60332b643",
            "name": "Zone A",
            "dwell_time": 60.01399612426758,
            "object_type": "universal_mail_container",
            "camera_id": "c211afbb-0608-4725-82c6-ffc540e29e02",
            "tmpfs_link": "/tmpfs/MAIN_1636457887205713000.jpg",
            "loc_x": 0.15444451570510864,
            "loc_y": 0.3650780916213989,
            "state": "open",
            "object_ids": [],
            "current_timestamp": 1636457887,
            "alert_type": "zone"
        }]}

    * **Error Response**
        ::

            HTTP/1.1 401 UnAuthorized
            Status: 401 UnAuthorized
            Content-Type: application/json
            Reasons:
            Provided access token is not valid.

    * **Error Response**
        ::

            HTTP/1.1 404 Alerts not Found
            Status: 404 Alerts not Found
            Content-Type: application/json
            Reasons:
            No Alert exist.
    """
    if not is_authenticated:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not check_resource_permissions(request=request):
        raise HTTPException(
            status_code=403,
            detail="You don't have enough permissions to access this resource.",
        )
    open_alerts = get_open_state_alerts()
    if open_alerts:
        return {"code": 200, "message": "success", "data": open_alerts}
    return {"code": 404, "message": "Alerts not found", "data": None}


@app.put("/alert-state", name="Update alert state")
async def update_alert_state(
        request: Request,
        alert_id: str = Form(...),
        is_authenticated: bool = Depends(authentication),
):
    """
    Update alert state to 'closed'

    * **URL**
        `localhost:8000/alert-state`

    * **Method**
        `[PUT]`

    * **Headers**
        * `'accept: application/json'`
        * `Authorization': 'bearer {token}'`

    * **Request Body**
        * `alert_id (Required)`

            `string`

    * **Response**
        ::

            HTTP/1.1 200 OK
            Date: Thu, 24 Feb 2011 12:36:30 GMT
            Status: 200 OK
            Content-Type: application/json
            Examples
            {"code": 200, "message": "success", "data": None}

    * **Error Response**
        ::

            HTTP/1.1 401 UnAuthorized
            Status: 401 UnAuthorized
            Content-Type: application/json
            Reasons:
            Provided access token is not valid.

    * **Error Response**
        ::

            HTTP/1.1 500 server error
            Status: 404 server error
            Content-Type: application/json
            Reasons:
            Cannot update alert state.
    """
    if not is_authenticated:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not check_resource_permissions(request=request):
        raise HTTPException(
            status_code=403,
            detail="You don't have enough permissions to access this resource.",
        )
    closed_alerts = change_alert_state(alert_id=alert_id)
    if closed_alerts:
        return {"code": 200, "message": "success", "data": None}
    return {"code": 500, "message": "server error"}


@app.delete("/delete-alert", name="Delete Alert")
async def delete_zone_alert(
        request: Request, alert_id: str, is_authenticated: bool = Depends(authentication)
):
    """
    Delete specific alert providing alert id

    * **URL**
        `localhost:8000/delete-alert`


    * **Method**
        `[DELETE]`

    * **Headers**
        * `'accept: application/json'`
        * `Authorization': 'bearer {token}'`

    * **Request Body**
        * `Alert ID (Required)`

            `string`

    * **Response**
        ::

            HTTP/1.1 200 OK
            Date: Thu, 24 Feb 2011 12:36:30 GMT
            Status: 200 OK
            Content-Type: application/json
            Examples
            {"message": "Success"}

    * **Error Response**
        ::

            HTTP/1.1 401 UnAuthorized
            Status: 401 UnAuthorized
            Content-Type: application/json
            Reasons:
            Provided access token is not valid.

    * **Error Response**
        ::

            HTTP/1.1 404 Alert Not Found
            Status: 404 Alert Not Found
            Content-Type: application/json
            Reasons:
            Camera not found in database.

    * **Error Response**
        ::

            HTTP/1.1 422 Validation Error
            Status: 422 Validation Error
            Content-Type: application/json
            Example:
            { "detail": [ { "loc": [ "string" ], "msg": "string", "type": "string" } ] }
    """
    if not is_authenticated:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not check_resource_permissions(request=request):
        raise HTTPException(
            status_code=403,
            detail="You don't have enough permissions to access this resource.",
        )
    zone_alert = delete_alerts(alert_id=alert_id, logging=logging)
    if zone_alert:
        return {"code": 200, "message": "success"}
    return {"code": 404, "message": "Alert Not Found"}


@app.get("/feature-control", name="Features")
async def get_feature_control(
        request: Request, is_authenticated: bool = Depends(authentication)
):
    """
    Decode feature control json from env variable

    * **URL**
        `localhost:8000/feature-control`

    * **Method**
        `[GET]`

    * **Headers**
        * `'accept: application/json'`
        * `Authorization': 'bearer {token}'`

    * **Response**
        ::

            HTTP/1.1 200 OK
            Date: Thu, 24 Feb 2011 12:36:30 GMT
            Status: 200 OK
            Content-Type: application/json
            Examples:
            {"code": 200, "message": "success", "data": data}

    * **Error Response**
        ::

            HTTP/1.1 401 UnAuthorized
            Status: 401 UnAuthorized
            Content-Type: application/json
            Reasons:
            Provided access token is not valid.

    * **Error Response**
        ::

            HTTP/1.1 404 Feature Control data not found.
            Status: 404 Not Found
            Content-Type: application/json
            Reasons:
            Feature Control data does not exist.
    """
    if not is_authenticated:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not check_resource_permissions(request=request):
        raise HTTPException(
            status_code=403,
            detail="You don't have enough permissions to access this resource.",
        )
    if data:
        return {"code": 200, "message": "success", "data": data}
    raise HTTPException(status_code=400, detail="Feature Control data not found.")


@app.post("/add-annotation", name="Add annotation")
async def add_annotation(
        request: Request,
        detections: str = Form(...),
        annotations: str = Form(...),
        image: str = Form(...),
        is_authenticated: bool = Depends(authentication),
):
    """
    Add annotation

    * **URL**
        `localhost:8000/add-annotation`


    * **Method**
        `[POST]`

    * **Headers**
        * `'accept: application/json'`
        * `Authorization': 'bearer {token}'`

    * **Request Body**
        * `Detections (Required)`

            `string`

        * `Annotations (Required)`

            `string`

        * `Image (Required)`

            `string`

    * **Response**
        ::

            HTTP/1.1 200 OK
            Date: Thu, 24 Feb 2011 12:36:30 GMT
            Status: 200 OK
            Content-Type: application/json
            Examples
            {"code": 200, "message": "success", "data": annotated_id}

    * **Error Response**
        ::

            HTTP/1.1 500 internal server error
            Status: 500 internal server error
            Content-Type: application/json
            Reasons:
            Annotation not inserted successfully.
    """
    if not is_authenticated:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not check_resource_permissions(request=request):
        raise HTTPException(
            status_code=403,
            detail="You don't have enough permissions to access this resource.",
        )
    annotation_image = {
        "annotations": json.loads(annotations),
        "detections": json.loads(detections),
        "image": image,
        "created_at": int(time.time()),
        "updated_at": int(time.time()),
        "archive_state": False,
    }
    annotated_id = annotation.insert_one(document=annotation_image).inserted_id
    if annotated_id:
        return {"code": 200, "message": "success", "data": {"id": str(annotated_id)}}
    raise HTTPException(status_code=500, detail="internal server error")


@app.put("/edit-annotation", name="Update annotation")
async def edit_annotation(
        request: Request,
        detections: str = Form(...),
        annotations: str = Form(...),
        image: str = Form(...),
        annotationId: str = Form(...),
        is_authenticated: bool = Depends(authentication),
):
    """
    Update annotation data against given id.

    * **URL**
        `localhost:8000/edit-annotation`


    * **Method**
        `[PUT]`

    * **Headers**
        * `'accept: application/json'`
        * `Authorization': 'bearer {token}'`

    * **Request Body**
        * `Detections (Required)`

            `string`

        * `Annotations (Required)`

            `string`

        * `Image (Required)`

            `string`

        * `Annotation Id (Required)`

            `string`

    * **Response**
        ::

            HTTP/1.1 200 OK
            Date: Thu, 24 Feb 2011 12:36:30 GMT
            Status: 200 OK
            Content-Type: application/json
            Examples:
            {"code": 200, "message": "success", "data": updated_annotations}

    * **Error Response**
        ::

            HTTP/1.1 500 internal server error
            Status: 500 internal server error
            Content-Type: application/json
            Reasons:
            Annotation not updated successfully.
    """
    if not is_authenticated:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not check_resource_permissions(request=request):
        raise HTTPException(
            status_code=403,
            detail="You don't have enough permissions to access this resource.",
        )
    annatation = annotation.find_one(filter=ObjectId(annotationId))
    annotation_image = {
        "detections": json.loads(detections),
        "annotations": json.loads(annotations),
        "image": image,
        "created_at": annatation.get("created_at"),
        "updated_at": int(time.time()),
        "archive_state": annatation.get("archive_state"),
    }
    annotation.replace_one(
        filter={"_id": ObjectId(annotationId)}, replacement=annotation_image
    )
    updated_data = annotation.find_one({"_id": ObjectId(annotationId)}, {"_id": 0})
    if updated_data:
        return {"code": 200, "message": "success", "data": updated_data}
    raise HTTPException(status_code=500, detail="internal server error")


@app.delete("/delete-annotation", name="Delete annotation")
async def delete_annotation(
        request: Request,
        annotationId: str,
        is_authenticated: bool = Depends(authentication),
):
    """
    Delete annotated data against given id.

    * **URL**
        `localhost:8000/delete-annotation`


    * **Method**
        `[DELETE]`

    * **Headers**
        * `'accept: application/json'`
        * `Authorization': 'bearer {token}'`

    * **Request Body**
        * `Annotation Id (Required)`

            `string`

    * **Response**
        ::

            HTTP/1.1 200 OK
            Date: Thu, 24 Feb 2011 12:36:30 GMT
            Status: 200 OK
            Content-Type: application/json
            Examples
            ```
            {"code": 200, "message": "success", "data": None}
            ```

    * **Error Response**
        ::

            HTTP/1.1 500 internal server error
            Status: 500 internal server error
            Content-Type: application/json
            Reasons:
            Annotation not deleted successfully.
    """
    if not is_authenticated:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not check_resource_permissions(request=request):
        raise HTTPException(
            status_code=403,
            detail="You don't have enough permissions to access this resource.",
        )
    if annotation.delete_one(filter={"_id": ObjectId(annotationId)}):
        return {"code": 200, "message": "success", "data": None}
    raise HTTPException(status_code=500, detail="internal server error")


@app.get("/get-annotation", name="List annotations")
async def get_annotation(
        request: Request,
        startTimestamp: int,
        endTimestamp: int,
        is_authenticated: bool = Depends(authentication),
):
    """
    List annotated data

    * **URL**
        `localhost:8000/get-annotation`


    * **Method**
        `[GET]`

    * **Headers**
        * `'accept: application/json'`
        * `Authorization': 'bearer {token}'`

    * **Request Body**
        * `Annotation Id (Required)`

            `string`

        * `startTimestamp (Required)`

            `integer`

        * `endTimestamp (Required)`

            `integer`

    * **Response**
        ::

            HTTP/1.1 200 OK
            Date: Thu, 24 Feb 2011 12:36:30 GMT
            Status: 200 OK
            Content-Type: application/json
            Examples:
            {"code": 200, "message": "success", "data": annotations}

    * **Error Response**
        ::

            HTTP/1.1 500 internal server error
            Status: 500 internal server error
            Content-Type: application/json
            Reasons:
            Failed to retrive the annotated list.
    """
    if not is_authenticated:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not check_resource_permissions(request=request):
        raise HTTPException(
            status_code=403,
            detail="You don't have enough permissions to access this resource.",
        )
    result = []
    if (startTimestamp and endTimestamp) == int(0):
        data = annotation.find({})
        for record in data:
            record["_id"] = str(record["_id"])
            result.append(record)
        return {"code": 200, "message": "success", "data": result}
    else:
        data = annotation.find(
            {"created_at": {"$gte": startTimestamp, "$lte": endTimestamp}}
        )
        for record in data:
            record["_id"] = str(record["_id"])
            result.append(record)
        return {"code": 200, "message": "success", "data": result}


@app.put("/archive-state", name="Update archive status")
async def update_archive_state(
        request: Request,
        annotationId: str,
        archiveStatus: bool,
        is_authenticated: bool = Depends(authentication),
):
    """
    Update archive status against the given id.

    * **URL**
        `localhost:8000/archive-state`

    * **Method**
        `[PUT]`

    * **Headers**
        * `'accept: application/json'`
        * `Authorization': 'bearer {token}'`

    * **Request Body**
        * `Annotation Id (Required)`

            `string`

    * **Response**
        ::

            HTTP/1.1 200 OK
            Date: Thu, 24 Feb 2011 12:36:30 GMT
            Status: 200 OK
            Content-Type: application/json
            Examples
            {"code": 200, "message": "success", "data": None}

    * **Error Response**
        ::

            HTTP/1.1 500 internal server error
            Status: 500 internal server error
            Content-Type: application/json
            Reasons:
            Failed to update the archive status.
    """
    if not is_authenticated:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not check_resource_permissions(request=request):
        raise HTTPException(
            status_code=403,
            detail="You don't have enough permissions to access this resource.",
        )
    archive_state = {"$set": {"archive_state": archiveStatus}}
    if annotation.update_one(
            filter={"_id": ObjectId(annotationId)}, update=archive_state
    ):
        return {"code": 200, "message": "success", "data": None}
    raise HTTPException(status_code=500, detail="internal server error")


@app.post("/annotation-feedback", name="Add annotation feedback")
async def add_annotation_feedback(
        request: Request,
        title: str = Form(...),
        message: str = Form(...),
        images: str = Form(...),
        is_authenticated: bool = Depends(authentication),
):
    """
    Add annotation feedback

    * **URL**
        `0.0.0.0:8000/annotation-feedback`


    * **Method**
        `[POST]`

    * **Headers**
        * `'accept: application/json'`
        * `Authorization': 'bearer {token}'`

    * **Request Body**
        * `Title (Required)`

            `string`

        * `Message (Required)`

            `string`

        * `Images (Required)`

            `string`

    * **Response**
            HTTP/1.1 200 OK
            Date: Thu, 24 Feb 2011 12:36:30 GMT
            Status: 200 OK
            Content-Type: application/json
            Examples
            ```
            {"code": 200, "message": "success", "data": feedback_id}
            ```

    * **Error Response**
            HTTP/1.1 500 internal server error
            Status: 500 internal server error
            Content-Type: application/json
            Reasons:
            Annotation feedback not inserted successfully.
    """
    if not is_authenticated:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not check_resource_permissions(request=request):
        raise HTTPException(
            status_code=403,
            detail="You don't have enough permissions to access this resource.",
        )
    annotation_feedback = {
        "title": title,
        "message": message,
        "images": images,
        "is_annotation": True,
        "created_at": int(time.time()),
        "updated_at": int(time.time()),
    }
    feedback_id = feedback.insert_one(document=annotation_feedback).inserted_id
    if feedback_id:
        return {"code": 200, "message": "success", "data": {"id": str(feedback_id)}}
    raise HTTPException(status_code=500, detail="internal server error")


@app.get("/annotation-get-classes", name="Get annotation classes")
async def get_annotation_classes(
        request: Request, is_authenticated: bool = Depends(authentication)
):
    """
    Get annotation classes

    * **URL**
        `localhost:8000/annotation-get-classes`


    * **Method**
        `[GET]`

    * **Headers**
        * `'accept: application/json'`
        * `Authorization': 'bearer {token}'`

    * **Request Body**
        * `Classes (Required)`

            `string`

    * **Response**
        ::

            HTTP/1.1 200 OK
            Date: Thu, 24 Feb 2011 12:36:30 GMT
            Status: 200 OK
            Content-Type: application/json
            Examples
            {"code": 200, "message": "success", "data": (classes))}

    * **Error Response**
        ::

            HTTP/1.1 500 internal server error
            Status: 500 internal server error
            Content-Type: application/json
            Reasons:
            Classes not available.
    """
    if not is_authenticated:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not check_resource_permissions(request=request):
        raise HTTPException(
            status_code=403,
            detail="You don't have enough permissions to access this resource.",
        )
    classes = ai_classes_col.find_one(
        {}, {"_id": 0}, sort=[("_id", pymongo.DESCENDING)]
    )["class_slug"]
    others = other_classes_col.find_one(
        {}, {"_id": 0}, sort=[("_id", pymongo.DESCENDING)]
    )
    if others:
        others = others["class_slug"]
    if others:
        classes = list(set(classes + others))
    classes.sort()
    if classes:
        return {"code": 200, "message": "success", "data": str(classes)}
    raise HTTPException(status_code=500, detail="internal server error")


@app.post("/annotation-add-class", name="Add other class")
async def set_annotation_others(
        request: Request,
        other: str = Form(...),
        is_authenticated: bool = Depends(authentication),
):
    """
    Add annotation class

    * **URL**
        `localhost:8000/annotation-add-class`


    * **Method**
        `[POST]`

    * **Headers**
        * `'accept: application/json'`
        * `Authorization': 'bearer {token}'`

    * **Request Body**
        * `Other (Required)`

            `string`

    * **Response**
        ::

            HTTP/1.1 200 OK
            Date: Thu, 24 Feb 2011 12:36:30 GMT
            Status: 200 OK
            Content-Type: application/json
            Examples
            {"code": 200, "message": "success", "data": (classes))}

    * **Error Response**
        ::

            HTTP/1.1 500 internal server error
            Status: 500 internal server error
            Content-Type: application/json
            Reasons:
            Classes not available.
    """
    if not is_authenticated:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not check_resource_permissions(request=request):
        raise HTTPException(
            status_code=403,
            detail="You don't have enough permissions to access this resource.",
        )
    others = other_classes_col.find_one(
        {}, {"_id": 0}, sort=[("_id", pymongo.DESCENDING)]
    )
    if others:
        others = others["class_slug"]
    if others:
        if other not in others:
            others.append(other)
            newothers = {"class_slug": others, "class_display_name": []}
            other_classes_col.insert_one(newothers)
    else:
        others = [other]
        newothers = {"class_slug": others, "class_display_name": []}
        other_classes_col.insert_one(newothers)

    if others:
        return {"code": 200, "message": "success", "data": str(others)}
    raise HTTPException(status_code=500, detail="internal server error")


@app.get("/annotation-get-keys", name="Get custom keys")
async def get_annotation_keys(
        request: Request, is_authenticated: bool = Depends(authentication)
):
    """
    Get annotation keys

    * **URL**
        `localhost:8000/annotation-get-keys`


    * **Method**
        `[GET]`

    * **Headers**
        * `'accept: application/json'`
        * `Authorization': 'bearer {token}'`

    * **Request Body**
        * `Classes (Required)`

            `string`

    * **Response**
        ::

            HTTP/1.1 200 OK
            Date: Thu, 24 Feb 2011 12:36:30 GMT
            Status: 200 OK
            Content-Type: application/json
            Examples
            {"code": 200, "message": "success", "data": (classes))}

    * **Error Response**
        ::

            HTTP/1.1 500 internal server error
            Status: 500 internal server error
            Content-Type: application/json
            Reasons:
            Classes not available.
    """
    if not is_authenticated:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not check_resource_permissions(request=request):
        raise HTTPException(
            status_code=403,
            detail="You don't have enough permissions to access this resource.",
        )
    keys = custom_keys_col.find_one({}, {"_id": 0}, sort=[("_id", pymongo.DESCENDING)])
    if keys:
        keys = keys["key_name"]
    if not keys:
        keys = []
    return {"code": 200, "message": "success", "data": str(keys)}


@app.post("/annotation-add-key", name="Add custom key")
async def set_annotation_keys(
        request: Request,
        key: str = Form(...),
        is_authenticated: bool = Depends(authentication),
):
    """
    Add annotation key

    * **URL**
        `localhost:8000/annotation-add-key`


    * **Method**
        `[POST]`

    * **Headers**
        * `'accept: application/json'`
        * `Authorization': 'bearer {token}'`

    * **Request Body**
        * `Key (Required)`

            `string`

    * **Response**
        ::

            HTTP/1.1 200 OK
            Date: Thu, 24 Feb 2011 12:36:30 GMT
            Status: 200 OK
            Content-Type: application/json
            Examples
            {"code": 200, "message": "success", "data": (keys))}

    * **Error Response**
        ::

            HTTP/1.1 500 internal server error
            Status: 500 internal server error
            Content-Type: application/json
            Reasons:
            Reasons:
            Classes not available.
    """
    if not is_authenticated:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not check_resource_permissions(request=request):
        raise HTTPException(
            status_code=403,
            detail="You don't have enough permissions to access this resource.",
        )
    keys = custom_keys_col.find_one({}, {"_id": 0}, sort=[("_id", pymongo.DESCENDING)])
    if keys:
        keys = keys["key_name"]
    if not keys:
        keys = list()
    if key not in keys:
        keys.append(key)
        keys.sort()
    custom_keys_col.insert_one({"key_name": keys})
    if keys:
        return {"code": 200, "message": "success", "data": str(keys)}
    raise HTTPException(status_code=500, detail="internal server error")


@app.post("/high-value-assets", name="Insert high value asset ids")
async def add_high_value_assets(
        request: Request,
        asset_ids: str = Form(...),
        is_authenticated: bool = Depends(authentication),
):
    """
    Insert high value object ids into the database.

    * **URL**
        `localhost:8000/high-value-assets`


    * **Method**
        `[POST]`

    * **Headers**
        * `'accept: application/json'`
        * `Authorization': 'bearer {token}'`

    * **Request Body**
        * `asset_ids (Required)`

            `string`

    * **Response**
        ::

            HTTP/1.1 200 OK
            Date: Thu, 24 Feb 2011 12:36:30 GMT
            Status: 200 OK
            Content-Type: application/json
            Examples
            {"code": 200, "message": "success", "data": None)}

    * **Error Response**
        ::

            HTTP/1.1 500 internal server error
            Status: 500 internal server error
            Content-Type: application/json
            Reasons:
            Data not inserted.
    """
    if not is_authenticated:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not check_resource_permissions(request=request):
        raise HTTPException(
            status_code=403,
            detail="You don't have enough permissions to access this resource.",
        )
    if insert_ids(ids=asset_ids, logging=logging):
        return {"code": 200, "message": "success", "data": None}
    raise HTTPException(status_code=500, detail="internal server error")


@app.get("/high-value-assets", name="Get high value asset ids")
async def get_high_value_assets(
        request: Request, is_authenticated: bool = Depends(authentication)
):
    """
    Get high value object ids from database.

    * **URL**
        `localhost:8000/high-value-assets`


    * **Method**
        `[POST]`

    * **Headers**
        * `'accept: application/json'`
        * `Authorization': 'bearer {token}'`


    * **Response**
        ::

            HTTP/1.1 200 OK
            Date: Thu, 24 Feb 2011 12:36:30 GMT
            Status: 200 OK
            Content-Type: application/json
            Examples
            {"code": 200, "message": "success", "data": None)}
    """
    if not is_authenticated:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not check_resource_permissions(request=request):
        raise HTTPException(
            status_code=403,
            detail="You don't have enough permissions to access this resource.",
        )
    data = get_ids(logging=logging)
    return {"code": 200, "message": "success", "data": data if data else {}}


@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Swagger UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="/static/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger-ui.css",
    )


@app.get(app.swagger_ui_oauth2_redirect_url, include_in_schema=False)
async def swagger_ui_redirect():
    return get_swagger_ui_oauth2_redirect_html()


@app.get("/redoc", include_in_schema=False)
async def redoc_html():
    return get_redoc_html(
        openapi_url=app.openapi_url,
        title=app.title + " - ReDoc",
        redoc_js_url="/static/redoc.standalone.js",
    )


@app.post("/feedback", name="Insert feedback for all pages")
async def add_feedback(
        request: Request,
        feedback_input: Feedback,
        is_authenticated: bool = Depends(authentication),
):
    """
    Insert feedback into the database.

    * **URL**
        `localhost:8000/feedback`


    * **Method**
        `[POST]`

    * **Headers**
        * `'accept: application/json'`
        * `Authorization': 'bearer {token}'`

    * **Request Body**
        * `feedback_input' (Required)`

            `string`

    * **Response**
        ::

            HTTP/1.1 200 OK
            Date: Thu, 24 Feb 2011 12:36:30 GMT
            Status: 200 OK
            Content-Type: application/json
            Examples
            {"code": 200, "message": "success", "data": None)}

    * **Error Response**
        ::

            HTTP/1.1 500 internal server error
            Status: 500 internal server error
            Content-Type: application/json
            Reasons:
            Data not inserted.
    """
    if not is_authenticated:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not check_resource_permissions(request=request):
        raise HTTPException(
            status_code=403,
            detail="You don't have enough permissions to access this resource.",
        )
    feedback_input = feedback_input.dict()
    if insert_feedback(feedback_table=feedback, feedback_input=feedback_input):
        return {"code": 200, "message": "success", "data": None}
    raise HTTPException(status_code=500, detail="internal server error")


@app.get("/feedback", name="Get feedback")
async def get_feedback_from_database(
        request: Request,
        is_annotation: bool = None,
        is_authenticated: bool = Depends(authentication),
):
    """
    Get feedback for annotation or any url.

    * **URL**
        `localhost:8000/feedback`


    * **Method**
        `[GET]`

    * **Headers**
        * `'accept: application/json'`
        * `Authorization': 'bearer {token}'`

    * **Request Body**
        * `is_annotation' (Optional)`

            `Boolean`

    * **Response**
        ::

            HTTP/1.1 200 OK
            Date: Thu, 24 Feb 2011 12:36:30 GMT
            Status: 200 OK
            Content-Type: application/json
            Examples
            {
                "code": 200,
                "message": "success",
                "data": {
                    "is_annotation": false,
                    "subject": "application",
                    "message": "abc",
                    "url": "www.google.com",
                    "created_at": 1626376068,
                    "updated_at": 1626376068
                })
            }

    * **Error Response**
        ::

            HTTP/1.1 500 internal server error
            Status: 500 internal server error
            Content-Type: application/json
            Reasons:
            Exception occurs while getting the feedback.
    """
    if not is_authenticated:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not check_resource_permissions(request=request):
        raise HTTPException(
            status_code=403,
            detail="You don't have enough permissions to access this resource.",
        )
    documents = get_feedback(is_annotation=is_annotation, feedback_table=feedback)
    if documents:
        return {"code": 200, "message": "success", "data": documents}
    raise HTTPException(status_code=500, detail="internal server error")


@app.get("/avgm-all", name="Get all avgm records")
async def get_all_avgm_from_oracle(
        request: Request, is_authenticated: bool = Depends(authentication)
):
    """
    Get all avgm rows.

    * **URL**
        `localhost:8000/avgm-all`


    * **Method**
        `[GET]`

    * **Headers**
        * `'accept: application/json'`
        * `Authorization': 'bearer {token}'`


    * **Response**
        ::

    * **Error Response**
        ::

            HTTP/1.1 500 internal server error
            Status: 500 internal server error
            Content-Type: application/json
            Reasons:
            Exception occurs while getting the avgm records.
    """
    if not is_authenticated:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not check_resource_permissions(request=request):
        raise HTTPException(
            status_code=403,
            detail="You don't have enough permissions to access this resource.",
        )
    avgm_records = get_all_avgm_records()
    if avgm_records:
        return {"code": 200, "message": "success", "data": avgm_records}
    raise HTTPException(status_code=404, detail="data not found")


@app.get("/avgm-mtel", name="Get avgm record against the given mtel_id")
async def get_avgm_record_by_key(
        request: Request, mtel_id: str, is_authenticated: bool = Depends(authentication)
):
    """
    Get avgm record against the given id..

    * **URL**
        `localhost:8000/avgm-mtel`


    * **Method**
        `[GET]`

    * **Headers**
        * `'accept: application/json'`
        * `Authorization': 'bearer {token}'`


    * **Response**
        ::

    * **Error Response**
        ::

            HTTP/1.1 500 internal server error
            Status: 500 internal server error
            Content-Type: application/json
            Reasons:
            Exception occurs while getting the avgm record.
    """
    if not is_authenticated:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not check_resource_permissions(request=request):
        raise HTTPException(
            status_code=403,
            detail="You don't have enough permissions to access this resource.",
        )
    raise HTTPException(status_code=404, detail="file not found")


@app.get("/mtel-ids", name="Get mtel ids")
async def get_mtel_ids_by_camera(
        request: Request, is_authenticated: bool = Depends(authentication)
):
    """
        Get all the mtel records

        * **URL**
            `localhost:8000/mtel-ids`


        * **Method**
            `[GET]`

        * **Headers**
            * `'accept: application/json'`
            * `Authorization': 'bearer {token}'`


        * **Response**
            ::
            HTTP/1.1 200 OK
                Date: Thu, 24 Feb 2011 12:36:30 GMT
                Status: 200 OK
                Content-Type: application/json
                Examples
                {
        "code": 200,
        "message": "success",
        "data": [
            {
                "camera_id": "7f9c72b9-0ba7-459f-be36-98bf0081e250",
                "mtel_ids": [
                    "99P990--1E1E2DP2W-44754322",
                    "99P990--1E1E2DP2W-44754343",
                    "99P990--1E1E2DW-44754350"
                ],
                "mtel_ids_count": 3
            },
            {
                "camera_id": "7f9c72b9-0ba7-459f-be36-98bf0081e500",
                "mtel_ids": [
                    "99P990--1E1E2DP2W-44754321",
                    "99P990--1E1E2DP2W-44754344"
                ],
                "mtel_ids_count": 2
            }
        ]
    }

        * **Error Response**
            ::

                HTTP/1.1 500 internal server error
                Status: 500 internal server error
                Content-Type: application/json
                Reasons:
                Exception occurs while getting the mtel_id records.
    """
    if not is_authenticated:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not check_resource_permissions(request=request):
        raise HTTPException(
            status_code=403,
            detail="You don't have enough permissions to access this resource.",
        )
    mtel_placards = get_mtel_ids_from_db()
    return {"code": 200, "message": "success", "data": mtel_placards}


@app.get("/object_details", name="Get object details")
async def get_object_details_from_classifiers(
        request: Request, is_authenticated: bool = Depends(authentication)
):
    """
                Get all objects and their possible fill levels and colors.

                * **URL**
                    `localhost:8000/object_details`


                * **Method**
                    `[GET]`

                * **Headers**
                    * `'accept: application/json'`
                    * `Authorization': 'bearer {token}'`


                * **Response**
                    ::
                    HTTP/1.1 200 OK
                        Date: Thu, 24 Feb 2011 12:36:30 GMT
                        Status: 200 OK
                        Content-Type: application/json
                        Examples
                        {
        "code": 200,
        "message": "success",
        "data": [
            {
                "class": "plastic_tray",
                "fill_levels": [
                    "empty",
                    "low",
                    "medium",
                    "high"
                ],
                "colors": [
                    "no_color",
                    "blue",
                    "orange",
                    "white"
                ]
            },
            {
                "class": "hamper_large_canvas",
                "fill_levels": [
                    "empty",
                    "low",
                    "medium",
                    "high"
                ],
                "colors": [
                    "no_color",
                    "blue",
                    "orange",
                    "white"
                ]
            },
            {
                "class": "fiberboard_tray",
                "fill_levels": [
                    "empty",
                    "low",
                    "medium",
                    "high"
                ],
                "colors": [
                    "no_color",
                    "blue",
                    "orange",
                    "white"
                ]
            }
        ]
    }

                * **Error Response**
                    ::

                        HTTP/1.1 500 internal server error
                        Status: 500 internal server error
                        Content-Type: application/json
                        Reasons:
                        Exception occurs while getting the object details.
    """
    if not is_authenticated:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not check_resource_permissions(request=request):
        raise HTTPException(
            status_code=403,
            detail="You don't have enough permissions to access this resource.",
        )
    object_details = get_object_details()
    if object_details:
        return {"code": 200, "message": "success", "data": get_object_details()}
    raise HTTPException(status_code=500, detail="internal server error")


@app.post(
    "/network-session", name="Scan the network and add a session into the database"
)
async def create_network_sessions(
        network_data: NetworkScan,
        background_task: BackgroundTasks,
        authorization: str = Header(None),
):
    """
    Create new session and check the online cameras.

            * **URL**
                `localhost:8000/network-session`

            * **Method**
                `[POST]`

            * **Headers**
                * `'accept: application/json'`
                * `Authorization': 'bearer {token}'`

            * **Response**
                ::
                HTTP/1.1 200 OK
                    Date: Thu, 24 Feb 2011 12:36:30 GMT
                    Status: 200 OK
                    Content-Type: application/json
                    Examples
            {
            "code": 200,
            "message": "success",
            "data": {
                "session_id": "613a04900bfed676b0386f17"
            }
        }
            ]
        }
        * **Error Response**
            ::

                HTTP/1.1 500 internal server error
                Status: 500 internal server error
                Content-Type: application/json
                Reasons:
                Exception occurs while creating the session.

                HTTP/1.1 400 invalid ip
                Status: 400 invalid ip
                Content-Type: application/json
                Reasons:
                Exception occurs if user enter invalid ip.
    """
    auth_user(authorization)
    network_data = network_data.dict()
    if validate_ip(ip=network_data.get("start_ip", "")) and validate_ip(
            ip=network_data.get("end_ip", "")
    ):
        session_id = create_network_session(network_data)
        background_task.add_task(
            active_cameras, network_data, session_id, token=authorization
        )
        if session_id:
            return {
                "code": 200,
                "message": "success",
                "data": {"session_id": session_id},
            }
        else:
            raise HTTPException(status_code=500, detail="internal server error")
    else:
        raise HTTPException(status_code=400, detail="invalid ip")


@app.get("/network-session", name="Get the network session")
async def retrive_network_session(authorization: str = Header(None)):
    """
        Get all the network session.

        * **URL**
            `localhost:8000/network-session`


        * **Method**
            `[GET]`

        * **Headers**
            * `'accept: application/json'`
            * `Authorization': 'bearer {token}'`

        * **Response**
            ::
            HTTP/1.1 200 OK
                Date: Thu, 24 Feb 2011 12:36:30 GMT
                Status: 200 OK
                Content-Type: application/json
                Examples
        {
                    "code": 200,
                    "message": "success",
                    "data": [
            {
                "_id": "6139f7988ac202e1f823d479",
                "status": "Processed",
                "created_at": 1631188888.2527952,
                "updated_at": 1631189062.4461489,
                "start_ip": "192.168.0.130",
                "end_ip": "192.168.0.135"
            },
            {
                "_id": "613a01e0c75ab6bd1b2600bd",
                "status": "Processed",
                "created_at": 1631191520.2643673,
                "updated_at": 1631191535.1267364,
                "start_ip": "192.168.0.130",
                "end_ip": "192.168.0.135"
            }
        ]
    }
        ]
    }
    """
    auth_user(authorization)
    network_session = get_network_session()
    return {
        "code": 200,
        "message": "success",
        "data": network_session if network_session else {},
    }


@app.get("/network-sensors", name="Get the network sensors")
async def retrive_network_sensors(session_id, authorization: str = Header(None)):
    """
        Get all the sensors that are on the same network.

        * **URL**
            `localhost:8000/network-sensors`


        * **Method**
            `[GET]`

        * **Headers**
            * `'accept: application/json'`
            * `Authorization': 'bearer {token}'`

        * **Request**
            * `session_id' (Required)`
              `String`

        * **Response**
            ::
            HTTP/1.1 200 OK
                Date: Thu, 24 Feb 2011 12:36:30 GMT
                Status: 200 OK
                Content-Type: application/json
                Examples
                {
        "code": 200,
        "message": "success",
        "data": [
            {
                "code": 200,
                "message": "success",
                "data": [
                    {
                        "_id": "613a04920bfed676b0386f18",
                        "session_id": "613a04900bfed676b0386f17",
                        "frame": "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAIBA
                        QEBAQIBAQECAgICAgQDAgICAgUEBAMEB",
                        "frame_width" : 1920,
                        "is_authenticated" : true,
                        "ip_address" : "192.168.0.136",
                        "user" : "admin",
                        "password" : "Darvis2020",
                        "camera_type" : "Network Camera",
                        "vendor" : ""
                    }
                        ]
            }
        ]
    }
    """
    auth_user(authorization)
    network_session = get_network_sensors(session_id=session_id)
    return {
        "code": 200,
        "message": "success",
        "data": network_session if network_session else {},
    }


@app.post("/bulk-camera", name="Add Cameras in bulk")
async def add_cameras_in_bulk(
        sensors_data: BulkSensors, authorization: str = Header(None)
):
    """
    Add cameras in bulk

    * **URL**
        `localhost:8000/bulk-camera`

    * **Method**
        `[POST]`

    * **Headers**
        * `'accept: application/json'`
        * `Authorization': 'bearer {token}'`

    * **Request Body**
        * `Camera Details  (Required)`

            `list`

        ::

            Example
                        [{
                "cameraId" : "5c8d2118-cb7a-4944-8da7-ee06159c2144",
                "cameraEnabled" : true,
                "displayName" : "Camera h",
                "url" : "192.168.1.136",
                "ip_address" : "192.168.1.136",
                "user" : "admin",
                "password" : "Darvis2020",
                "protocol" : "rtsp",
                "homography" : false,
                "src_points" : [],
                "dst_points" : [],
                "frameWidth" : 1920,
                "frameHeight" : 1080,
                "frames_per_second" : 1,
                "createdAt" : ISODate("2021-09-16T08:16:13.685Z"),
                "image" : "weqfdyqewfdcjqwekgdcqwe
                jdcfgwqejhfdcewuhjyfdchjewdcjehwvgdcjuhewf",
                "cone_dst_points" : [],
                "cone_src_points" : [],
                "cameraIconConfig" : {
                    "x" : 0,
                    "y" : 0,
                    "rotation" : 0
                },
                "cameraConeStatus" : false,
                "CameraConfigOrientation" : {
                    "x" : 0,
                    "y" : 0,
                    "z" : 0,
                    "yaw" : 0,
                    "pitch" : 0
                }
            }]

    * **Response**
        ::

            HTTP/1.1 200 OK
            Date: Thu, 24 Feb 2011 12:36:30 GMT
            Status: 200 OK
            Content-Type: application/json
            Examples
            {"message": "Camera Added successfully",
            "data": ["fd17e2e9-56cd-4907-ba48-104343b09a2c"]}

    * **Error Response**
        ::

            HTTP/1.1 401 UnAuthorized
            Status: 401 UnAuthorized
            Content-Type: application/json
            Reasons:
            Provided access token is not valid.

    * **Error Response**
        ::

            HTTP/1.1 409 Conflict
            Status: 409 Already added
            Content-Type: application/json
            Example:
            {"code": 409,"message": "Already added","data": []}
    """
    auth_user(authorization)
    response = insert_bulk_of_cameras(sensors_data=sensors_data)
    if response["code"] == 200:
        return {
            "code": 200,
            "message": "Camera Added successfully",
            "data": response["camera_ids"],
        }
    return {"code": 409, "message": "Already added", "data": response["camera_ids"]}


@app.post("/authenticate-camera", name="Authenticate camera")
async def authenticate_camera(
        auth_sensor: AuthenticateSensor, authorization: str = Header(None)
):
    """
    Authenticate the given camera.

    * **URL**
        `localhost:8000/authenticate-camera`

    * **Method**
        `[POST]`

    * **Headers**
        * `'accept: application/json'`
        * `Authorization': 'bearer {token}'`

    * **Request Body**
        * `Camera Details  (Required)`

            `dict`

        ::

            Example
            {
                "ip_address": "192.168.0.136",
                "user": "admin",
                "password": "Darvis2020",
                "url": "192.168.0.136",
                "session_id": "61407d591f0324a7b220092f",
                "camera_id":"61407da51f0324a7b2200930"
            }

    * **Response**
        ::

            HTTP/1.1 200 OK
            Date: Thu, 24 Feb 2011 12:36:30 GMT
            Status: 200 OK
            Content-Type: application/json
            Examples
            {
                    "code": 200,
                    "message": "Camera authenticated successfully"
                }

    * **Error Response**
        ::

            HTTP/1.1 401 UnAuthorized camera
            Status: 401 UnAuthorized camera
            Content-Type: application/json
            Examples:
                        {
            "code": 401,
            "message": "Bad username or password"
        }
            Provided credentials are not valid.

    * **Error Response**
        ::

            HTTP/1.1 500 internal server error
            Status: 500 internal server error
            Content-Type: application/json
            Example:
            {
            "code": 500,
            "message": "internal server error"
        }
    """
    auth_user(authorization)
    status = camera_authentication(sensor_data=auth_sensor, token=authorization)
    if status == 200:
        return {
            "code": 200,
            "message": "Camera authenticated successfully",
        }
    elif status == 401:
        return {
            "code": 401,
            "message": "Bad username or password",
        }
    else:
        return {
            "code": 500,
            "message": "internal server error",
        }


@app.delete("/delete-session", name="Delete the network session")
async def delete_network_session(session_id: str, authorization: str = Header(None)):
    """
    Delete the network session against the given session id.

    * **URL**
        `localhost:8000/delete-session`


    * **Method**
        `[DELETE]`

    * **Headers**
        * `'accept: application/json'`
        * `Authorization': 'bearer {token}'`

    * **Response**
        ::
        HTTP/1.1 200 OK
            Date: Thu, 24 Feb 2011 12:36:30 GMT
            Status: 200 OK
            Content-Type: application/json
            Examples
               {
            "code": 200,
            "message": "success"
        }

        * **Error Response**
    ::

        HTTP/1.1 500 internal server error
        Status: 500 internal server error
        Content-Type: application/json
        Example:
        {
        "code": 500,
        "message": "internal server error"
    }
    """
    auth_user(authorization)
    status = delete_session(session_id=session_id)
    if status:
        return {"code": 200, "message": "success"}
    return {"code": 500, "message": "internal server error"}
